# การจำแนกยีนมะเร็งจาก gene expression โดยใช้ backpropagation multilayer perceptron


### Dataset
-------------
gene expression cancer RNA-Seq Data Set เป็นชุดข้อมูลเกี่ยวกับ gene expression ที่แสดงออกถึงมะเร็งชนิดต่างๆ ได้แก่ BRCA, KIRC, COAD, LUAD และ PRAD ซึ่งแต่ละ instance คือ RNA sequence โดยที่แต่ละ attribute คือ ยีนแต่ละชนิดที่แตกต่างกัน
gene expression cancer RNA-Seq Data Set เป็นไฟล์ csv ภายในมี 801 instances,  20,531 attribute  และไม่มี missing value โดยที่ labels ของ class จะแยกเป็นอีกไฟล์หนึ่ง ชื่อ labels.csv เผยแพร่วันที่ 9 เดือน มิถุนายน ปี 2016 นำรายละเอียดข้อมูลมาจาก Samuele Fiorini, samuele.fiorini '@' dibris.unige.it, University of Genoa, [ UCI Machine Learning Repository: gene expression cancer RNA-Seq Data Set ] (เข้าถึงข้อมูลเมื่อ 10/02/2022)
ที่มา : https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

### Preprocessing
-------------
ภายใน dataset ชุดนี้ไม่มี missing value ดังนั้นขั้นตอนแรกที่จะทำเลยก็คือการกำจัด noise ที่เกิดจาก bias ของข้อมูลแต่ละ attribute ซึ่งมี range ของตัวเลขที่ไม่ใช่ช่วงเดียวกัน (เนื่องจาก dataset ชุดนี้มี 20,531 attribute ดังนั้นผู้เขียนจะยกตัวอย่างแค่บาง attribute ที่ทำให้เห็นภาพชัดเจนมากที่สุด) 

![](/blob/raw_data.jpeg)

จากภาพในการสุ่ม gene (attribute) มาดูข้อมูลภายในจะเห็นว่า range ของตัวเลขในแต่ละ gene ไม่ได้อยู่ช่วงเดียวกัน ทำให้เวลา train ข้อมูลจาก back propagation จะเกิด bias ขึ้น ทำให้ประสิทธิภาพของผลลัพธ์แย่ลง ดังนั้นผู้เขียนจึงทำให้ข้อมูลอยู่ในช่วง range เดียวกัน โดยการทำ Normalization ซึ่งเป็นชนิด Z-scores Normalization 

![](/blob/norm_data.jpeg)

จากภาพเป็นข้อมูลที่ผ่านการทำ Z-scores Normalization มาแล้ว จะเห็นว่าข้อมูลทุก attribute จะอยู่ใน range เดียวกัน เนื่องจากมองจากตาเปล่าอาจเห็นภาพไม่ชัด ดังนั้นผู้เขียนจะทำให้เห็นช่วงของข้อมูลได้ดีมากขึ้นดังนี้

![](/blob/norm_1.png)

![](/blob/norm_2.png)

![](/blob/norm_3.png)

หลังจากที่ preprocessing ตัวข้อมูลที่เป็น input ไปแล้ว เราก็ต้องทำการ preprocessing ในฝั่งของ output ด้วย (ภาพข้างล่างเป็นแค่ส่วนหนึ่งของ dataset เท่านั้น ไม่ใช่ข้อมูลทั้งหมด)

![](/blob/labels.jpeg)

จากภาพเป็น labels ที่บอกถึงมะเร็งชนิดต่าง ๆ ของ gene expression แต่ละแบบ ซึ่งหากจะนำข้อมูลนี้ไปคำนวณก็ต้องนำข้อมูลเหล่านี้ไปเข้ารหัสเพื่อเปลี่ยนให้เป็นตัวเลขก่อน ดังนี้

![](/blob/class.jpeg)

![](/blob/categorical.jpeg)

จากนั้นเราจะนำตัวเลขเหล่านี้ไปเข้ารหัสอีกครั้งหนึ่ง ซึ่งเรียกว่า one hot vector ซึ่งเป็นการเปลี่ยนตัวเลขให้เป็น vector 1 หน่วย ที่จะเป็นตัวแทนของคลาสนั้นๆ ดังภาพ

![](/blob/preprocessing.jpeg)

ภาพนี้จะเป็นกระบวนการเข้ารหัสของ labels ซึ่งจะนำไปคำนวณในช่วงของ error ของ neural network (ภาพดังกล่าวเป็นแค่ส่วนหนึ่งของ dataset เท่านั้น ไม่ใช่ทั้งหมดของ dataset)

หลังจากที่ preprocessing เสร็จแล้วขั้นตอนสุดท้ายก็คือการทำ shuffle data หรือก็คือการสลับตำแหน่งของข้อมูลแบบสุ่ม เพื่อไม่ให้ในส่วนของโครงสร้างของ neural network จดจำแค่ข้อมูลที่นำไปเทรนด์แต่ต้อง robust กับข้อมูลด้วย (ในส่วนของขั้นตอนนี้ ไม่มีภาพการแสดงผลให้เห็น เนื่องจากตัวชุดข้อมูลที่ดาวน์โหลดมาก่อนแล้วนั้น ไม่ได้เรียงข้อมูลตามคลาสมาก่อน[สลับมาก่อนแล้ว] ทำให้ถ้าแสดงผลแล้วอาจทำให้ดูไม่ต่างจากเดิม)

### Structure
-------------
