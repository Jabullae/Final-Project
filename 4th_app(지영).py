from flask import Flask, render_template, request, flash, redirect, url_for
from werkzeug.utils import secure_filename
#import sqlite3 as sql
import os
import random

app = Flask(__name__)
app.secret_key = '다음'


# html 렌더링
@app.route('/',  methods=['POST','GET'])
def wrong_img():
    # 랜덤으로 텍스트 보내기
    count = 0
    random_class = ['나비','지렁이','컴퓨터']
    random.shuffle(random_class)
    for i in random_class :
        random_list = [i+'1', i+'2',i+'3',i+'X']
        random_list_2 = []
        random.shuffle(random_list)
        for  j in random_list:
            random_list_2.append(j)
        img1 = random_list_2[0]
        img2 = random_list_2[1]
        img3 = random_list_2[2]
        img4 = random_list_2[3]
        
        
        
    
    
        
    
   
    # 누른 버튼의 text 를 받아서 정답인지 오답인지 판별하기
    point =[]
    if request.method == 'POST':
        image = str(request.form['button'])
        if 'X' in image:
            point.append('정답')
        else: point.append('오답')
        count += 1 
    if count == 3:
            
        msg = '다음'    
    
        
    return render_template('4th_test.html',img1 = img1, img2=img2,img3=img3,img4=img4,count = count)


@app.route('/',  methods=['POST','GET'])
def end():
    'NEXT'
    return render_template('4th_test.html')


    

if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug=True)  