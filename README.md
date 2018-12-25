# Question-Generation-Pytorch
Question Generation model implementation in pytorch

Pytorch는 Define-by-Run 방식으로 유연한 딥러닝 프레임워크를 제공하여, 많은 이들의 사랑을 받고 있다. 이번 글에서는 Torch와 TensorFlow 로는 구현되어 있으나, 아직 Pytorch 로 구현되 있지 않은 Du et al.,( 2017, ACL ) <https://arxiv.org/pdf/1705.00106.pdf> 을 구현해보고자 한다.

Du et al.,( 2017 ).Learning to Ask : Neural Question Generation for Reading Comprehension


Result
Sentence1 : johann eck , speaking on behalf of the empire as assistant of the archbishop of trier , presented luther with copies of his writings laid out on a table and asked him if the books were his , and whether he stood by their contents . 

Question1 : who was the father of UNK UNK

Sentence2 : he prayed , consulted friends , and gave his response the next day : 
Question2 : what was the name of the person that worked for the first high school in the 1980s UNK

Sentence3 : at the end of this speech , luther raised his arm `` in the traditional salute of a knight winning a bout . ''
Question3 : where was the republica cafe found in the UNK palace of UNK

Sentence4 : the largest of these is the eldon square shopping centre , one of the largest city centre shopping complexes in the uk .
Question4 : what is the largest lake in london UNK

Sentence5 : inflammation is one of the first responses of the immune system to infection .
Question5 : what is the first system to use the system called UNK

