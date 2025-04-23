if [ "$1" = "clean" ]; then
    python -m code.utils.models delete --type trace_classifier
fi

python -m code.models.training.trace_classifier --name trace-classifier-betavae128 --autoencoder h-betavae128 -e 30 -t 0
python -m code.models.training.trace_classifier --name trace-classifier-betavae128-alpha --autoencoder h-betavae128 -e 30 -t 0 -a

python -m code.models.training.trace_classifier --name trace-classifier-betavae128-mt --autoencoder h-betavae128 -e 30 -t 2
python -m code.models.training.trace_classifier --name trace-classifier-betavae128-mt-alpha --autoencoder h-betavae128 -e 30 -t 2 -a