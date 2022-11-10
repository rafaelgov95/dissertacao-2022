import tensorflow as tf

tf.keras.backend.set_learning_phase(0)  # Ignore dropout at inference
model = tf.keras.models.load_model('/home/rafael/Documentos/DSP/RR/Statistica/CNN-HRV/best_model.h5')
export_path = './tensorflow_models/vfc-rv/1'

with tf.keras.backend.get_session() as sess:
    tf.saved_model.simple_save(
        sess,
        export_path,
        inputs={'sensor': model.input},
        outputs={t.name: t for t in model.outputs}) 

