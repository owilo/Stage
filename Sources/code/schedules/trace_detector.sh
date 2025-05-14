if [ "$1" = "clean" ]; then
    python -m code.utils.models delete --type trace_detector
fi

# python -m code.models.training.trace_detector --name trace-detector-betavae128 --autoencoder h-betavae128 -e 20 -t 0
# python -m code.models.training.trace_detector --name trace-detector-betavae128-alpha --autoencoder h-betavae128 -e 20 -t 0 -a

# python -m code.models.training.trace_detector --name trace-detector-betavae128-std --autoencoder h-betavae128 -e 20 -t 1
# python -m code.models.training.trace_detector --name trace-detector-betavae128-std-alpha --autoencoder h-betavae128 -e 20 -t 1 -a

# python -m code.models.training.trace_detector --name trace-detector-betavae128-mt --autoencoder h-betavae128 -e 20 -t 2
# python -m code.models.training.trace_detector --name trace-detector-betavae128-mt-alpha --autoencoder h-betavae128 -e 20 -t 2 -a

# python -m code.models.training.trace_detector --name trace-detector-betavae128-d --autoencoder h-betavae128 -e 20 -t 1 --encode
# python -m code.models.training.trace_detector --name trace-detector-betavae128-mt-d --autoencoder h-betavae128 -e 20 -t 2 --encode

# python -m code.models.training.trace_detector --name trace-detector-cvae128 --autoencoder h-cvae128 -e 20
python -m code.models.training.trace_detector --name trace-detector-cvae128-d --autoencoder h-cvae128 -e 20 --encode