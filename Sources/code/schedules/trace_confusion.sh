# python -m code.analytics.trace_confusion --name "betavae" -t 0 --autoencoder "h-betavae128" --classifier "classifier" --tdetector "trace-detector-betavae128" --tclassifier "trace-classifier-betavae128"
# python -m code.analytics.trace_confusion --name "betavae-alpha" -t 0 -a --autoencoder "h-betavae128" --classifier "classifier" --tdetector "trace-detector-betavae128-alpha" --tclassifier "trace-classifier-betavae128-alpha"

# python -m code.analytics.trace_confusion --name "betavae-mt" -t 2 --autoencoder "h-betavae128" --classifier "classifier" --tdetector "trace-detector-betavae128-mt" --tclassifier "trace-classifier-betavae128-mt"
# python -m code.analytics.trace_confusion --name "betavae-mt-alpha" -t 2 -a --autoencoder "h-betavae128" --classifier "classifier" --tdetector "trace-detector-betavae128-mt-alpha" --tclassifier "trace-classifier-betavae128-mt-alpha"

# python -m code.analytics.trace_confusion --name "betavae-d" -t 0 --autoencoder "h-betavae128" --classifier "classifier" --tdetector "trace-detector-betavae128-d" --tclassifier "trace-classifier-betavae128" --encode
# python -m code.analytics.trace_confusion --name "betavae-mt-d" -t 2 --autoencoder "h-betavae128" --classifier "classifier" --tdetector "trace-detector-betavae128-mt-d" --tclassifier "trace-classifier-betavae128-mt" --encode

python -m code.analytics.trace_confusion --name "cvae" -t 0 --autoencoder "h-cvae128" --classifier "classifier" --tdetector "trace-detector-cvae128" --tclassifier "trace-classifier-cvae128"
python -m code.analytics.trace_confusion --name "cvae-d" -t 0 --autoencoder "h-cvae128" --classifier "classifier" --tdetector "trace-detector-cvae128-d" --tclassifier "trace-classifier-cvae128" --encode