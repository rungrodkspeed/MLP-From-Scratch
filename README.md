# การจำแนกยีนมะเร็งจาก gene expression โดยใช้ backpropagation multilayer perceptron


### Dataset
-------------
gene expression cancer RNA-Seq Data Set เป็นชุดข้อมูลเกี่ยวกับ gene expression ที่แสดงออกถึงมะเร็งชนิดต่างๆ ได้แก่ BRCA, KIRC, COAD, LUAD และ PRAD ซึ่งแต่ละ instance คือ RNA sequence โดยที่แต่ละ attribute คือ ยีนแต่ละชนิดที่แตกต่างกัน
gene expression cancer RNA-Seq Data Set เป็นไฟล์ csv ภายในมี 801 instances,  20,531 attribute  และไม่มี missing value โดยที่ labels ของ class จะแยกเป็นอีกไฟล์หนึ่ง ชื่อ labels.csv เผยแพร่วันที่ 9 เดือน มิถุนายน ปี 2016 นำรายละเอียดข้อมูลมาจาก Samuele Fiorini, samuele.fiorini '@' dibris.unige.it, University of Genoa, [ UCI Machine Learning Repository: gene expression cancer RNA-Seq Data Set ] (เข้าถึงข้อมูลเมื่อ 10/02/2022)
ที่มา : https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq

### Preprocessing
-------------
ภายใน dataset ชุดนี้ไม่มี missing value ดังนั้นขั้นตอนแรกที่จะทำเลยก็คือการกำจัด noise ที่เกิดจาก bias ของข้อมูลแต่ละ attribute ซึ่งมี range ของตัวเลขที่ไม่ใช่ช่วงเดียวกัน (เนื่องจาก dataset ชุดนี้มี 20,531 attribute ดังนั้นผู้เขียนจะยกตัวอย่างแค่บาง attribute ที่ทำให้เห็นภาพชัดเจนมากที่สุด) 

<p align="center">
  <img src="/blob/raw_data.jpg" />
</p>

จากภาพในการสุ่ม gene (attribute) มาดูข้อมูลภายในจะเห็นว่า range ของตัวเลขในแต่ละ gene ไม่ได้อยู่ช่วงเดียวกัน ทำให้เวลา train ข้อมูลจาก back propagation จะเกิด bias ขึ้น ทำให้ประสิทธิภาพของผลลัพธ์แย่ลง ดังนั้นผู้เขียนจึงทำให้ข้อมูลอยู่ในช่วง range เดียวกัน โดยการทำ Normalization ซึ่งเป็นชนิด Z-scores Normalization 

![](/blob/norm_data.jpg)

จากภาพเป็นข้อมูลที่ผ่านการทำ Z-scores Normalization มาแล้ว จะเห็นว่าข้อมูลทุก attribute จะอยู่ใน range เดียวกัน เนื่องจากมองจากตาเปล่าอาจเห็นภาพไม่ชัด ดังนั้นผู้เขียนจะทำให้เห็นช่วงของข้อมูลได้ดีมากขึ้นดังนี้

![](/blob/norm_1.png)

![](/blob/norm_2.png)

![](/blob/norm_3.png)

หลังจากที่ preprocessing ตัวข้อมูลที่เป็น input ไปแล้ว เราก็ต้องทำการ preprocessing ในฝั่งของ output ด้วย (ภาพข้างล่างเป็นแค่ส่วนหนึ่งของ dataset เท่านั้น ไม่ใช่ข้อมูลทั้งหมด)

![](/blob/labels.jpg)

จากภาพเป็น labels ที่บอกถึงมะเร็งชนิดต่าง ๆ ของ gene expression แต่ละแบบ ซึ่งหากจะนำข้อมูลนี้ไปคำนวณก็ต้องนำข้อมูลเหล่านี้ไปเข้ารหัสเพื่อเปลี่ยนให้เป็นตัวเลขก่อน ดังนี้

![](/blob/class.jpg)

![](/blob/categorical.jpg)

จากนั้นเราจะนำตัวเลขเหล่านี้ไปเข้ารหัสอีกครั้งหนึ่ง ซึ่งเรียกว่า one hot vector ซึ่งเป็นการเปลี่ยนตัวเลขให้เป็น vector 1 หน่วย ที่จะเป็นตัวแทนของคลาสนั้นๆ ดังภาพ

![](/blob/preprocessing.jpg)

ภาพนี้จะเป็นกระบวนการเข้ารหัสของ labels ซึ่งจะนำไปคำนวณในช่วงของ error ของ neural network (ภาพดังกล่าวเป็นแค่ส่วนหนึ่งของ dataset เท่านั้น ไม่ใช่ทั้งหมดของ dataset)

หลังจากที่ preprocessing เสร็จแล้วขั้นตอนสุดท้ายก็คือการทำ shuffle data หรือก็คือการสลับตำแหน่งของข้อมูลแบบสุ่ม เพื่อไม่ให้ในส่วนของโครงสร้างของ neural network จดจำแค่ข้อมูลที่นำไปเทรนด์แต่ต้อง robust กับข้อมูลด้วย (ในส่วนของขั้นตอนนี้ ไม่มีภาพการแสดงผลให้เห็น เนื่องจากตัวชุดข้อมูลที่ดาวน์โหลดมาก่อนแล้วนั้น ไม่ได้เรียงข้อมูลตามคลาสมาก่อน[สลับมาก่อนแล้ว] ทำให้ถ้าแสดงผลแล้วอาจทำให้ดูไม่ต่างจากเดิม)

### Structure
-------------
- จำนวน features = 20,531 ดังนั้น node ของ input ก็จะเท่ากับ 20,531 ด้วย
- จำนวน Class = 5 ดังนั้น node ของ output ก็จะเท่ากับ 5 ด้วย
- Activation function ของ hidden layer จะใช้ sigmoid แต่ในส่วนของ output จะใช้เป็น softmax เนื่องจากเป็น function ที่ใช้สำหรับ classification หลายคลาส และใช้ cost function คือ cross entropy 
- hidden layer 1 ชั้น และมีจำนวน neurons เท่ากับ 128

เมื่อสร้าง Neural Network ตามข้อดังกล่าวจะได้โครงสร้างออกมาดังนี้

![](/blob/structure.jpeg)

จากนั้นเราจะกำหนด hyperparameter ดังนี้

- learning rate = 0.1
- momentum rate = 0.55
- จำนวน epochs = 10

### Cross Validation
-------------
Cross Validation หรือ การทดสอบแบบไขว้ เป็นการแบ่งข้อมูลแบบสุ่ม ออกเป็นจำนวนกลุ่มตามตัวเลข คือถ้าใส่ตัวเลข ค่า K เข้าไปจะมีการแบ่งข้อมูล ออกเป็นส่วนย่อย ๆ จำนวน K และจะเก็บข้อมูลจำานวน 1 ส่วนย่อย เพื่อไว้ ทดสอบส่วนข้อมูลที่เหลือ จะนำามาสร้างโมเดลและจะทำาวนไปจนกว่าข้อมูลถูกใช้ ทดสอบจนหมดทุกส่วน

ซึ่งในการทำครั้งนี้เราจะใช้ 10-folds cross validation ดังภาพ

![](/blob/10_folds.jpg)

ซึ่งส่วนที่ถูกระบายสีนั้นเป็นส่วนของ testing set ดังนั้นจากภาพนี้จะเห็นว่า เราจะต้องสร้าง model 10 แบบ เพื่อเทรนด์ข้อมูลที่มีลักษณะแตกต่างกันในแต่ละแถว

### Result
-------------
หลังจากที่ออกแบบตัว model เสร็จ เราก็จะนำตัว model ไปเทรนด์ข้อมูลในแต่ละแถว ซึ่งทุกครั้งที่เปลี่ยนแถวของข้อมูลใหม่ เราก็จะประกาศตัว model ใหม่ ทำให้มี model ทั้งหมด 10 แบบ แล้วจะได้ผลลัพธ์ออกมาดังนี้

![](/blob/bin_folds.jpg)

![](/blob/acc_table.PNG)

ต่อไปก็เป็นผลลัพธ์ที่แสดงถึง confusion matrix ดังนี้
