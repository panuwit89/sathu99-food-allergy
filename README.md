# Food Allergy Chatbout for NLP course
> [!IMPORTANT]
> อย่าลืมใส่ Channel_access_token="" ในไฟล์ app/Config.py

## ติดตั้ง requirements.txt ก่อน (กรณีไม่ได้รันใน docker)
```
pip install -r requirements.txt
```

## ขั้นตอนการ Deploy
1. สร้าง Project ใหม่ใน Google Cloud
2. `cd <path_project>`
3. ใช้คำสั่ง `gcloud auth configure-docker`
4. ใช้คำสั่ง `docker build -t <name_image> .`
5. ใช้คำสั่ง `docker run -d -p <number_port>:<number_port> <name_image>`
6. ใช้คำสั่ง `docker run -d -p <number_port>:<number_port> <name_image>`
7. ใช้คำสั่ง `docker push <location_push>:<tag_name>`

> [!NOTE]
> <location_push> คือ Path ตําแหน่งที่เราจะ Push ขึ้นไป โดยมีรูปแบบเช่น grc.io/<ไอดีของโปรเจคที่เราสร้างใน google cloud>/<ชื่อโฟลเดอร์ที่เราอยากใส่>