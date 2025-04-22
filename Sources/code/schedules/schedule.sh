#python -m code.models.training.trace_classifier --name trace-classifier-betavae16 --autoencoder h-betavae16
#python -m code.models.training.trace_detector --name trace-detector-betavae16 --autoencoder h-betavae16

#python -m code.models.training.trace_classifier --name trace-classifier-betavae128-std --autoencoder h-betavae128
#python -m code.models.training.trace_detector --name trace-detector-betavae128-std --autoencoder h-betavae128 -e 20

# python -m code.models.training.trace_classifier --name trace-classifier-betavae128-mt --autoencoder h-betavae128 -e 30
# python -m code.models.training.trace_detector --name trace-detector-betavae128-mt --autoencoder h-betavae128 -e 20

# python -m code.models.training.trace_classifier --name trace-classifier-betavae128-mt-alpha --autoencoder h-betavae128 -e 30 -a
# python -m code.models.training.trace_detector --name trace-detector-betavae128-mt-alpha --autoencoder h-betavae128 -e 20 -a

# python -m code.models.training.trace_classifier --name trace-classifier-betavae128-alpha --autoencoder h-betavae128 -e 30 -a
# python -m code.models.training.trace_detector --name trace-detector-betavae128-alpha --autoencoder h-betavae128 -e 20 -a

python -m code.utils.models delete --type trace_detector
python -m code.models.training.trace_detector --name trace-detector-betavae128 --autoencoder h-betavae128 -e 20 -t 0
python -m code.models.training.trace_detector --name trace-detector-betavae128-alpha --autoencoder h-betavae128 -e 20 -t 0 -a

python -m code.models.training.trace_detector --name trace-detector-betavae128-mt --autoencoder h-betavae128 -e 20 -t 2
python -m code.models.training.trace_detector --name trace-detector-betavae128-mt-alpha --autoencoder h-betavae128 -e 20 -t 2 -a