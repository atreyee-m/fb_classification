#!/bin/bash



mv  Data\&Data\ Classification\ Challenge\ \-\ Facebook\ \-\ Training\ Set.csv fb_cls_tr.csv;
sed -i -e '1d' fb_cls_tr.csv;
cut -f1 -d$'\t' fb_cls_tr.csv > train.txt ;
cut -f9 -d$'\t' fb_cls_tr.csv > target.txt;
python fb_classification.py


