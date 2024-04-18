import pandas as pd
import streamlit as st
import qiskit
import pylatexenc
import ipywidgets
import qiskit_aer
import numpy as np
import scipy
import matplotlib.pyplot as plt
import sympy
import sklearn
DS = pd.read_csv('StressLevelDataset.csv')


st.title('STUDENT STRESS PREDICTION ')
st.sidebar.header('STUDENT DATA')

st.subheader('DEPRESSION LEVEL')


fig, ax = plt.subplots(figsize=(5,3))
mental_health_history_chart = DS['depression'].value_counts().plot(kind='bar')

plt.xlabel('Depression level')
plt.ylabel('Count')




# Pass the figure explicitly to st.pyplot()
st.pyplot(mental_health_history_chart.figure,fig)
st.write('1-4  Minimal Depression')
st.write('5-9  Mild Depression')
st.write('10-14  Moderate Depression')
st.write('15-19  Moderately Severe Depression')
st.write('20-27 Severe  Depression')



X = DS.iloc[: ,0:20].values
Y = DS.iloc[: , 20].values

from sklearn.model_selection import train_test_split as tts

X_train, X_test, Y_train, Y_test = tts(X, Y, test_size = 0.2, random_state = 100)

from sklearn.preprocessing import StandardScaler as SS
SC = SS()

X_train = SC.fit_transform(X_train)
X_test = SC.transform(X_test)

def user_report():

  anxiety_level = st.sidebar.slider('State your Anxiety Level', 0,21, 3 )
  self_esteem = st.sidebar.slider('How would you rate your Self-Esteem?', 0,30, 1 )
  mental_health_history = st.sidebar.slider('Do you have a history of mental health issues?', 0,1, 0 )
  depression = st.sidebar.slider('State your Depression Level', 0,27, 2 )
  headache = st.sidebar.slider(' How often do you experience headaches?', 0,5, 1 )
  blood_pressure = st.sidebar.slider('How would you rate your blood pressure?', 1,3, 0 )
  sleep_quality = st.sidebar.slider('Kindly rate your Sleep Quality', 0,5, 0 )
  breathing_problem = st.sidebar.slider('Did you face any Breathing Problem due to Stress?', 0,5, 3 )
  noise_level = st.sidebar.slider('How would you rate the noise level in your environment?', 0, 5, 3)
  living_conditions = st.sidebar.slider(' How satisfied are you with your living conditions?', 0, 5, 3)
  basic_needs = st.sidebar.slider('How satisfied are you with your access to basic needs (food, shelter, etc.)?', 0, 5, 3)
  safety= st.sidebar.slider('How safe do you feel in your environment?', 0,5, 3 )
  academic_performance= st.sidebar.slider('How would you rate your Academic Performance?', 0,5, 3 )
  study_load= st.sidebar.slider('How would you rate your Study Load?', 0,5, 3 )
  teacher_student_relationship= st.sidebar.slider('How would you rate your relationship with your teachers?', 0,5, 3 )
  future_career_concerns= st.sidebar.slider('How concerned are you about your future career prospects?', 0,5, 3 )
  social_support= st.sidebar.slider(' How much social support do you perceive to have?', 0,3, 1 )
  peer_pressure= st.sidebar.slider('How much peer pressure do you experience?', 0,5, 3 )
  extracurricular_activiities= st.sidebar.slider('How many times a week you pratice Extracurricular Activiities', 0,5, 3 )
  bullying= st.sidebar.slider('How often do you experience bullying?', 0,5, 3 )


  user_report = {
      'anxiety_level': anxiety_level,
      'self_esteem': self_esteem,
      'mental_health_historymental_health_history': mental_health_history,
      'depression': depression,
      'headache': headache,
      'blood_pressure': blood_pressure,
      'noise_level': noise_level,
      'living_conditions': living_conditions,
      'basic_needs': basic_needs,
      'sleep_quality': sleep_quality,
      'breathing_problem': breathing_problem,
      'safety': safety,
      'academic_performance': academic_performance,
      'study_load': study_load,
      'teacher_student_relationship': teacher_student_relationship,
      'future_career_concerns': future_career_concerns,
      'social_support': social_support,
      'peer_pressure': peer_pressure,
      'extracurricular_activiities': extracurricular_activiities,
      'bullying': bullying,

  }

  report_data = pd.DataFrame(user_report, index=[0])
  return report_data

user_data=user_report()


from qiskit_aer import  Aer
from qiskit import  QuantumCircuit
from qiskit import transpile
from math import sqrt
from sklearn.metrics import recall_score, precision_score, confusion_matrix

def pqc_classify(backend, passenger_state):
    """backend -- a qiskit backend to run the quantum circuit at
    passenger_state -- a valid quantum state vector"""

    # Create a quantum circuit with one qubit
    qc = QuantumCircuit(2)

    # Define state |Psi> and initialize the circuit
    qc.initialize(passenger_state, 0)

    # Measure the qubit
    qc.measure_all()

    transpiled_qc = transpile(qc, backend)

    # run the transpiled quantum circuit
    result = backend.run(transpiled_qc).result()

    # get the counts, these are either {'0': 1} or {'1': 1}
    counts=result.get_counts(qc)

    # get the bit 0 or 1
    return int(list(map(lambda item: item[0], counts.items()))[0])

def run(f_classify, x):
    return list(map(f_classify, x))

def specificity(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[0][1]) if (matrix[0][0]+matrix[0][1] > 0) else 0

def npv(matrix):
    return matrix[0][0]/(matrix[0][0]+matrix[1][0]) if (matrix[0][0]+matrix[1][0] > 0) else 0
def classifier_report(name, run, classify, input, labels):
    cr_predictions = run(classify, input)
    cr_cm = confusion_matrix(labels, cr_predictions)

    cr_precision = precision_score(labels, cr_predictions)
    cr_recall = recall_score(labels, cr_predictions)
    cr_specificity = specificity(cr_cm)
    cr_npv = npv(cr_cm)
    cr_level = 0.25*(cr_precision + cr_recall + cr_specificity + cr_npv)


def predict_stress():
    user_data = user_report()


backend = Aer.get_backend('statevector_simulator')
#backend = Aer.get_backend('qasm_simulator')

# Specify the quantum state that results in either 0 or 1
initial_state = [1/sqrt(2), 1/sqrt(2)]

classifier_report("Random PQC",
    run,
    lambda passenger: pqc_classify(backend, initial_state),
    X_train,
    Y_train)
# Specify the quantum state for prediction (modify as needed)
prediction_state = [1/sqrt(2), 1/sqrt(2)]

# Make predictions on the entire test set
predictions = run(lambda passenger: pqc_classify(backend, initial_state), user_data)



st.header('Your Report: ')
output=''
if predictions[0]==0:
  output = 'You have no stress'
else:
  output = ' You have stress'
st.subheader(output)
#st.write(output)
