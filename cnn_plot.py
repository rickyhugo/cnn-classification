import visualkeras
from tensorflow import keras

# Create a new model instance
mdl = keras.models.load_model('best_weights')
mdl.summary()
visualkeras.layered_view(mdl, legend=True, to_file="model_vis.png")

if __name__ == '__main__':
    print('Running...')