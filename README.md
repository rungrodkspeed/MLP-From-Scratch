# การจำแนกยีนมะเร็งจาก gene expression โดยใช้ backpropagation multilayer perceptron


### Dataset
-------------
gene expression cancer RNA-Seq Data Set เป็นชุดข้อมูลเกี่ยวกับ gene expression ที่แสดงออกถึงมะเร็งชนิดต่างๆ ได้แก่ BRCA, KIRC, COAD, LUAD และ PRAD ซึ่งแต่ละ instance คือ RNA sequence โดยที่แต่ละ attribute คือ ยีนแต่ละชนิดที่แตกต่างกัน
gene expression cancer RNA-Seq Data Set เป็นไฟล์ csv ภายในมี 801 instances,  20,531 attribute  และไม่มี missing value โดยที่ labels ของ class จะแยกเป็นอีกไฟล์หนึ่ง ชื่อ labels.csv เผยแพร่วันที่ 9 เดือน มิถุนายน ปี 2016 นำรายละเอียดข้อมูลมาจาก Samuele Fiorini, samuele.fiorini '@' dibris.unige.it, University of Genoa, [ UCI Machine Learning Repository: gene expression cancer RNA-Seq Data Set ] (เข้าถึงข้อมูลเมื่อ 10/02/2022)
ที่มา : https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

### Preprocessing
-------------
