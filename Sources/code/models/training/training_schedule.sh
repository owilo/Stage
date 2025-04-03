#python -m code.models.training.trace_classifier --name trace-classifier-betavae16 --autoencoder h-betavae16
#python -m code.models.training.trace_detector --name trace-detector-betavae16 --autoencoder h-betavae16

python -m code.models.training.trace_classifier --name trace-classifier-betavae128-std --autoencoder h-betavae128
python -m code.models.training.trace_detector --name trace-detector-betavae128-std --autoencoder h-betavae128