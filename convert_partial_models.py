import setGPU
import tensorflow
from models import dense_embedding
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model
import numpy as np
import hls4ml
import pandas as pd
from qkeras.utils import _add_supported_quantized_objects
from models import dense_embedding, dense_embedding_quantized
co = {}
_add_supported_quantized_objects(co)


def print_dict(d, indent=0):
    align = 20
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent+1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))


# load full model:
# model = tensorflow.keras.models.load_model('output/model.h5', compile=False, custom_objects=co)
# prepare new model:
n_puppi_cands = 16
reuse_factor = 1
precision = 'ap_fixed<8,3>'
io_type = 'io_parallel'
strategy = 'Resource'
output_dir = 'hls_output_{}_{}_rf{}_puppi{}'.format(io_type, strategy, reuse_factor, n_puppi_cands)
batch_size = 1
model = dense_embedding_quantized(n_features=6,
                                  n_features_cat=2,
                                  number_of_pupcandis=n_puppi_cands,
                                  embedding_input_dim={0: 13, 1: 3},
                                  emb_out_dim=8,
                                  with_bias=False,
                                  t_mode=1,
                                  units=[12, 36],
                                  logit_total_bits=8,
                                  logit_int_bits=2,
                                  activation_total_bits=8,
                                  activation_int_bits=2)
# load just weights:
# model.load_weights('output/model.h5')

# check everthing works
model.summary()
model.save('{}/model.h5'.format(output_dir))

# now let's break up the model
# save a dictionary of the model
model_dict = {layer.name: layer for layer in model.layers}

# first part of the model
partial_model_1 = Model(inputs=[model_dict['input_cont'].input, model_dict['input_cat0'].input, model_dict['input_cat1'].input],
                        outputs=model_dict['met_weight_minus_one'].output, name='partial_model_1')
partial_model_1.summary()

# second part of the model
input_layer_1 = Input(model_dict['input_pxpy'].input.shape[1:])
input_layer_2 = Input(model_dict['met_weight_minus_one'].output.shape[1:])
x = model_dict['multiply']([input_layer_2, input_layer_1])
output_layer = model_dict['output'](x)
partial_model_2 = Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer, name='partial_model_2')
partial_model_2.summary()

partial_model_1.save('{}/partial_model_1.h5'.format(output_dir))
partial_model_2.save('{}/partial_model_2.h5'.format(output_dir))

# let's check the partial models give the same answer as the original model
# random inputs
X = np.random.rand(batch_size, n_puppi_cands, 4)
Xp = np.random.rand(batch_size, n_puppi_cands, 2)
X_cat0 = np.random.randint(13, size=(batch_size, n_puppi_cands))
X_cat1 = np.random.randint(3, size=(batch_size, n_puppi_cands))
print('input shapes')
print(X.shape)
print(Xp.shape)
print(X_cat0.shape)
print(X_cat1.shape)

# original output
y = model.predict([X, Xp, X_cat0, X_cat1])
print('original output')
print(y)

# partial ouputs
y_1 = partial_model_1.predict([X, X_cat0, X_cat1])
y_2 = partial_model_2.predict([Xp, y_1])
print('partial outputs')
print(y_2)

# check if they're equal (error if they're not!)
np.testing.assert_array_equal(y, y_2)

model_to_convert = partial_model_1
config = hls4ml.utils.config_from_keras_model(model_to_convert, granularity='name',
                                              default_reuse_factor=reuse_factor, default_precision=precision)
config['Model']['Strategy'] = strategy
config['LayerName']['input_cat0']['Precision']['result'] = 'ap_uint<4>'
config['LayerName']['input_cat1']['Precision']['result'] = 'ap_uint<4>'
# skip optimize_pointwise_conv
# config['SkipOptimizers'] = ['optimize_pointwise_conv']
# for layer in config['LayerName'].keys():
#    config['LayerName'][layer]['Trace'] = True

print("-----------------------------------")
print_dict(config)
print("-----------------------------------")
hls_model = hls4ml.converters.convert_from_keras_model(model_to_convert,
                                                       hls_config=config,
                                                       io_type=io_type,
                                                       output_dir=output_dir,
                                                       part='xcvu9p-flgb2104-2-i',
                                                       clock_period=5)
hls_model.compile()

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file='{}/model_hls4ml.png'.format(output_dir))

y_1_hls = hls_model.predict([X.astype(np.float32), X_cat0.astype(np.float32), X_cat1.astype(np.float32)])
df = pd.DataFrame({'keras': y_1.flatten(), 'hls4ml': y_1_hls.flatten()})
print(df)


hls_model.build(synth=True)
hls4ml.report.read_vivado_report(output_dir)
