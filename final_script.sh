#!/bin/bash
######################################################################################
#atremukh@silo [/nobackup/atremukh/PycharmProjects/codebase/fb_classification]# cat target.txt | grep "Fake Seller" | wc -l
#9174
#atremukh@silo [/nobackup/atremukh/PycharmProjects/codebase/fb_classification]# cat target.txt | grep "Reseller" | wc -l
#9583
#atremukh@silo [/nobackup/atremukh/PycharmProjects/codebase/fb_classification]# cat target.txt | grep "No Seller" | wc -l
#16425

######################################################################################

mv  Data\&Data\ Classification\ Challenge\ \-\ Facebook\ \-\ Training\ Set.csv fb_cls_tr.csv;
sed -i -e '1d' fb_cls_tr.csv;
cut -f1 -d$'\t' fb_cls_tr.csv > train.txt ;
cut -f9 -d$'\t' fb_cls_tr.csv > target.txt;
python fb_classification.py


