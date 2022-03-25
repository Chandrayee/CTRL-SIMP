# CTRL-SIMP

CTRL (Controllable) SIMP (Text simplification) is medical text simplification model that enables users to identify words or phrases within the medical text that they find difficult and would like to be removed, replaced or elaborated. The model is released with the first ever medical dataset with fine-grained annotations of elaboration, replacement, deletion and insertion. The model is trained in a multi-angle fashion like MACAW (https://github.com/allenai/macaw), so that, in addition to simplifying the medical text and can also separately predict the words that are difficult foe people to understand. 

CTRL-SIMP is built on top of T5, similar to MACAW. Unlike MACAW the training data has both inputs split into slots and inputs with inplace annotations.
