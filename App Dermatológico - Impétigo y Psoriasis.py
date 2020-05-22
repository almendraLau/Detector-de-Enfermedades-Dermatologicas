import streamlit as st 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input

files0 = 'images/psoriasis.jpg'
files1 = 'images/impetigo.jpg'
files2 = 'images/no.jpg'
files3 = 'images/logo.jpg'
files4 = 'images/infografia.jpg'

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

def cnn ():
  MODEL_FILE = 'model/filename.model'
  model = load_model(MODEL_FILE)
  return model

app_mode = st.sidebar.selectbox("", ["INICIO", "PROTOCOLO", "APLICACIÓN"])
if app_mode == 'INICIO':
  imagen = Image.open(files3)
  st.image(imagen, use_column_width=True)
  st.write("> Una Aplicación de Machine Learning para el reconocimiento de dos enfermedades de la piel: _Impétigo_ y _Psoriasis_")
  infogra = Image.open(files4)
  st.image(infogra, use_column_width=True)
  st.markdown("<div align='justify'>"
        "En el ámbito mundial, según una investigación del Centro del Cáncer de la Universidad de Colorado (2017), se señala que las enfermedades de la piel ocupan el cuarto lugar como causa de discapacidad que se desarrolla a lo largo de la vida de una persona. "
        "Dentro del grupo de enfermedades que se destacan principalmente en la población infantil y adulta del Perú, se tiene al impétigo y psoriasis, sin embargo, existen otras como la sarna, urticarias, entre otros. "
        "</div><br>", unsafe_allow_html=True)

  st.markdown("<div align='justify'>"
  "Este proyecto pretende proporcionar una herramienta tecnológica orientada a la reducción de brechas relacionadas al diagnóstico"
  " médico de enfermedades de la piel de la población de la ciudad de Iquitos. En el contexto del problema se ha tomado en"
  " cuenta el alto número de pacientes con problemas o enfermedades de la piel, bajo número de personas que asisten a los centros"
  " de salud u hospitales, así como el reducido número de atenciones a los males de la piel y la escasa cultura que posee la población" 
  "respecto a visitar a los médicos en los centros de salud."
  "</div><br>", unsafe_allow_html=True)

  st.subheader("💡 ¿Sabías que..?")
  st.success("Según un reportaje realizado por el Instituto Nacional de Radio y Televisión (2019), se tiene una gran cifra de 400, 000 personas padeciendo de psoriasis en el Perú, considerando que a nivel mundial afecta a más de 125 millones de personas. Cabe decir, que la psoriasis es considerada por la Organización mundial de la Salud (2015) como una enfermedad muy dolorosa, desfigurante e incapacitante. ")
  st.header("📚 SOBRE LAS ENFERMEDADES")
  st.subheader("Impetigo")
  st.markdown("<div align='justify'>"
    "El Impétigo es una infección de la piel común y muy contagiosa que afecta principalmente a bebés y niños."
    " El impétigo normalmente aparece en forma de llagas rojas en la cara, especialmente alrededor de la nariz y "
    "la boca, y en las manos y los pies. Las llagas revientan y producen costras color miel.(Fuente: MayoClinic)"
    "</div><br>", unsafe_allow_html=True)
  imagen = Image.open(files1)
  st.image(imagen, caption="Fuente: Mayo Clinic", use_column_width=True)
  st.subheader("Psoriasis")
  st.markdown("<div align='justify'>"
    "La psoriasis es una enfermedad frecuente de la piel que acelera el ciclo de vida de las células cutáneas. "
    "Hace que las células se acumulen rápidamente en la superficie de la piel. Las células cutáneas excedentes forman"
    " escamas y manchas rojas que causan comezón y, a veces, dolor.(Fuente: MayoClinic)"
    "</div><br>", unsafe_allow_html=True)
  imagen = Image.open(files0)
  st.image(imagen, caption="Fuente: Fundacion Piel Sana", use_column_width=True)




elif app_mode=='PROTOCOLO':
  
  st.title("🔰 PROTOCOLO DE CAPTURA")
  st.write(
    '> Antes de hacer uso de la Aplicación Machine Learning, es importante conocer el _Protocolo de Captura_ de las imágenes.'
    )
  st.write(
    "Se hizo una investigación para determinar el siguiente protocolo para tomar las fotografías, de esta manera obtenemos"
    " una mejor precisión al hacer uso de la Aplicación Machine Learning; por ello se les remienda hacer uso de estas."
  )
  st.subheader("Requisitos de la Cámara Fotográfica:")
  st.markdown("* La cámara debe tener una resolución de 13Mp a más.")
  st.markdown("* Modo de Cámara: Normal.")
  st.subheader("Respecto a la Iluminación:")
  st.markdown("* La zona afectada, en lo posible debe estar iluminada por luz natural.")
  st.markdown("* No usar el flash de la camará para tomar las fotografías.")
  st.subheader("Otras Consideraciones:")
  st.markdown("* La fotografía debe contener más del 60% de la zona afectada.")
  st.markdown("* Se recomienda una distancia entre la cámara y la piel, de unos 10 a 15cm aproximadamente.")
  st.subheader("¿Cómo NO debemos tomar las fotografías?")
  imagen = Image.open(files2)
  st.image(imagen, caption="Fuente: Elaboración Propia", use_column_width=True)
  

elif app_mode=='APLICACIÓN':

  st.title("APLICACIÓN DETECTOR DE IMPÉTIGO Y PSORIASIS®")
  st.markdown("<div align='justify'>Esta es una aplicación de la demostración del Algoritmo. El Algoritmo puede identificar 2 tipos"
  " de enfermedades de la piel: PSORIASIS e IMPÉTIGO, mediante una fotografia de la zona infectada. El Algoritmo fue construido usando"
  " la Red Neuronal Convolucional Inception V3."
  "</div>", unsafe_allow_html=True)
  st.header("🚀 EMPECEMOS")
  st.markdown("<div align='justify'>Sí usted sospecha la presencia de una de estas enfermedades en su piel, elija la fotografía tomada"
  " desde su explorador de archivos o arrastrelo con el mouse y el Algoritmo se encargará de hacer la detección."
  "</div>", unsafe_allow_html=True
  )
  
  files = st.file_uploader("Formato de imagen: JPG", type="jpg")
  
  @st.cache
  def predict(model, img):
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return preds[0]
      
  if files is not None:

    imagen = Image.open(files)
    st.image(imagen, use_column_width=True)
    
    st.header("🎯RESULTADOS")
    st.markdown(" >⏳ Tiempo de Ejecución: _50 - 60s_")
    
    img = image.load_img(files)
    ancho= 500
    wpercent = (ancho/float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((ancho, hsize), Image.ANTIALIAS)
    preds = predict(cnn(), img)

    labels = ("IMPÉTIGO", "PSORIASIS")
    plt.figure(figsize=(8,4))
    plt.xlabel("Probabilidad (%)")
    plt.barh([0, 1], preds*100, alpha=0.8)

    for i, v in enumerate(np.round(preds*100,1)):
      plt.text(v+0.05,i -0.1, str(v) + "%", color='r', fontsize=13)
            
    plt.yticks([0, 1], labels)
    plt.xlim(0, 112)
    st.pyplot()

    st.markdown("**Recuerde que el médico será el ercagado de determinar el diágnostico con una mayor precisión.**")


st.sidebar.header("Resumen")
st.sidebar.info("La Aplicación ML, es un software capaz de detectar 2 enfermedades de la piel, basado en Inteligencia Artificial,"
" en la que se usa la CNN INCEPTION V3. Además se aplicó las técnicas de Transfer Learning y Data Augmentation, para el manejo y"
" aumento de datos. Como resultado de la investigación, se obtuvo resultados alentadores al tener una precisión  total del"
" 84% al hacer inferencia de datos.")
st.sidebar.markdown("Investigador: _Alejandro Reátegui Pezo_")