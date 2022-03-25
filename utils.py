DEFAULT_SLOT_FORMAT = {
    "slot": "$SLOT$",
    "assign": " = ",
    "separator": " ; ",
    "missing_value": "N/A"
}
SLOT_SHORTFORMS = {"S": "Simple", "E": "Expert", "X": "Elaborate", "Xs": "Add sentence", "C": "Context", "D": "Delete", "R": "Replace", "F":"Define", "Ea":"Annotated expert", "Sa":"Annotated Simple"}
ANNOTATION_SLOT_MAP = {"<elab>": "X", "<del>": "D", "<rep>": "R", "<elab-define>":"F", "<elab-sentence>": "Xs"}
SLOT_FROM_LC = {x.lower(): x for x in SLOT_SHORTFORMS.values()}
SLOT_KEY_FROM_LC = {v.lower(): k for k, v in SLOT_SHORTFORMS.items()}
#we will use at the most 5 output slots: SXFRD or SaXFR
#we will use at the most 4 input slots: EXFRD or EaXFR
#thinking to leave the deletion for the model to learn

#more on decoder options: https://huggingface.co/blog/how-to-generate
GENERATOR_OPTIONS_DEFAULT = {"min_length": 1, "max_length": 128, "num_beams": 1, "num_return_sequences": 1,
                             "do_sample": False, "top_k": 50, "top_p": 1.0, "temperature": 1.0,
                             "length_penalty": 1.0, "repetition_penalty": 1.0}
                          #explain the generator options, is num_beams=1 good? what is top_p and top_k, beam_size 5 or 10
                          #beam search is good enough when generating text of predictable length
                          #beam search doesn't follow human text generation, there is no surprise (unless trained explicitly to create surprise)
                          #top-p is for nucleus sampling, considers smallest number of words with p > cumulative density 