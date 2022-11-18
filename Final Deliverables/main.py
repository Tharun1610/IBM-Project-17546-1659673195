import numpy as np
import os
from tensorflow.python.keras.models import load_model,model_from_json
from keras_preprocessing import image
import tensorflow as tf
global graph
graph=tf.compat.v1.get_default_graph()
from flask import Flask,request,render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

app=Flask(__name__,template_folder='static')
@app.route('/')
def index():
    return render_template("index.html")


@app.route('/predict',methods = ['GET','POST'])
def upload():
    if request.method == 'POST':
        f = request.files['image']
        print("current path")
        basepath = os.path.dirname(__file__)
        print("current path", basepath)
        filepath = os.path.join(basepath,'uploads',f.filename)
        print("upload folder is ", filepath)
        f.save(filepath)
        
        img = image.load_img(filepath,target_size = (224,224))
        x = image.img_to_array(img)
        x = np.expand_dims(x,axis =0)
        
        with graph.as_default():
            jf=open("animal1.json")
            lmj=jf.read()
            loaded_model=model_from_json(lmj)
            loaded_model.load_weights("animal1.h5")
            preds=np.argmax(loaded_model.predict(x.reshape(1, 224,224,3)),axis=1)
                        
            print("prediction",preds)
            
        Prediction_category=["TheGreatIndianBustard or Indian bustard, is a bustard found on the Indian subcontinent. A large bird with a horizontal body and long bare legs, giving it an ostrich like appearance, this bird is among the heaviest of the flying birds.",
                            "The spoon-billed sandpiper (Calidris pygmaea) is a small wader which breeds on the coasts of the Bering Sea and winters in Southeast Asia. This species is highly threatened, and it is said that since the 1970s the breeding population has decreased significantly. By 2000, the estimated breeding population of the species was 350–500.",
                            "The corpse flower (Amorphophallus titanum) also known as titan arum, reeks of rotting flesh and death when in bloom. Lucky for us, this stinky plant blooms once every seven to nine years according to the Eden Project(opens in new tab) and each bloom only lasts 24 to 36 hours",
                            "Lady’s slipper orchids are usually terrestrial, though some are epiphytic or grow on rocks. Most species have rhizomes and fibrous roots. Unlike most other orchids, the flowers characteristically feature two fertile anthers (male, pollen-producing structures) instead of just one. The slipper-shaped lip of the flower serves as a trap for pollinating insects, forcing insect visitors to climb past the reproductive structures and deposit or receive pollinia (pollen masses) to fertilize the flower",
                            "The Seneca white deer are a rare herd of deer living within the confines of the former Seneca Army Depot in Seneca County, New York. When the 10,600-acre (43 km2) depot was created in 1941, a 24-mile (39 km) fence was erected around its perimeter, isolating a small herd of white-tailed deer, some of which had white coats",
                            "Pangolins, sometimes known as scaly anteaters, are mammals of the order Pholidota. The one extant family, the Manidae, has three genera: Manis, Phataginus, and Smutsia. Manis comprises the four species found in Asia, while Phataginus and Smutsia include two species each, all found in sub-Saharan Africa."
                            ]
        
        text=Prediction_category[preds[0]]
    return text
if __name__ == '__main__':
    app.run(debug = True, threaded = False,port="8000")
