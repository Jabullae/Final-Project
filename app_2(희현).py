from flask import Flask, render_template, request, flash, g, redirect
from werkzeug.utils import secure_filename
import sqlite3 as sql
import os

# 1st test
import keras.applications as kapp
import keras.preprocessing.image as kimage
import keras.models as kmodels
import numpy as np
import keras.utils as utils
from anchor import *

# 2nd test
# 3rd test
import pandas
import torch
import sqlite3

# 4th test
import random

# 5th test
import base64
import requests
from time import sleep
import urllib3
import json
import os, random, string
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import easyocr
import pyscreenshot

# 6st test


 # flask name 선언
app = Flask(__name__)


# HTML 렌더링
################### 홈페이지 ###################
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/intro')
def intro():
    return render_template('0_intro.html')


################### 1번째 게임 : 유사도게임 ###################

vgg_model = kapp.VGG16(weights='imagenet', include_top=False)
model = kmodels.Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block5_pool').output)

def get_image_feature(img_path):
    img = utils.load_img(img_path, target_size=(224, 224))
    img = utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = kapp.vgg16.preprocess_input(img)
    features = model.predict(img)
    features = features.flatten()
    return features

@app.route("/vgg")
def similarity_image():
    q, p_path, h_path, sim = random_sim()
    return render_template('1st_test_2.html', q=q, p_path=p_path, h_path=h_path, sim=sim)

# @app.route("/sim_test", methods=["POST"])
# def sim_test():
#     p_path = str(request.form['p_path'])
#     sim = float(request.form['sim'])
#     return render_template('sim_test.html', p_path=p_path, sim=sim)

@app.route("/image-similarity", methods=["POST"])
def image_similarity():
    f = request.files['file']
    img_path = 'C:/Users/admin/Desktop/SoftLand/static/1/img/anchor/img.jpg'
    f.save(img_path)
    p_path = str(request.form['p_path'])
    sim = float(request.form['sim'])

    features1 = get_image_feature(p_path)
    features2 = get_image_feature(img_path)

    cosine_similarity = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    print(cosine_similarity)
    if cosine_similarity >= sim:
        return render_template('success.html')
    else:
        return render_template('fail.html')


################### 2번째 게임 : 스트루프 ###################
@app.route('/stroop')
def stroop():
    return render_template('2nd_test.html')


################### 3번째 게임 : 글->그림 ###################
@app.route('/text_to_img')
def text_to_img():
    return render_template('3rd_test.html')

@app.route("/", methods=["GET", "POST"])
def predict():
    # Model(YOLOv5 종속 항목 설치)
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = 'best.pt', force_reload =True)
    # Image
    img = ['C:\\Users\\admin\\Desktop\\최종 프로젝트\\글,그림\\벤치.png']
    # 그림판에그림을 그려서 어떻게 여기에 갖고올지 생각... img 변수에 담기
    ########## 이 사진을 어떻게 가지고 올지에 대해서 알아봐야한다. !!

    # 추론
    results = model(img)


    # 결과
    #results.print()
    #results.show()
    #results.save() # Save image to 'runs\detect\exp'
    #results.xyxy[0]  # 예측 (tensor)
    # results.pandas().xyxy[0]  # 예측 (pandas)
    conf = results.pandas().xyxy[0]
   
    # 오답 여부
    OX = []
    if str(conf.name) == '토끼':
        OX.append('정답')
    else : OX.append('오답')
    print(OX)
    

    # DB 생성 / 이미 있으면 나중에 주석처리하기.
    # isolation_level = None (auto commit)
    conn = sqlite3.connect('ijm.db', isolation_level=None)
    # 커서
    cursor = conn.cursor()
    # 테이블 생성(데이터 타입 = TEST, NUMERIC, INTEGER, REAL, BLOB(image) 등)
    # 필드명(ex. name) -> 데이터 타입(ex. text) 순서로 입력 
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS text_write (
        id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
        game text,
        point float,
        OX text)""")

    # db 에 정보 저장
    game = '글->그림'
    point = float(conf.confidence)
    OX = OX[0]

    cursor.execute("""
        INSERT INTO text_write (game, point, OX) VALUES (?,?,?)          
        """, (game, point, OX)
        )

    conn.commit()
    cursor.close()
    conn.close()    

    
    render_template('index.html')
    


@app.route('/find_diff',  methods=['POST','GET'])
def wrong_img():
    # 랜덤으로 텍스트 보내기

    ClickCount = 0
    
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
        
    ClickCount = ClickCount + 1    
    print(ClickCount)
        
    if ClickCount == 3:
        return render_template('5th_test.html')
    else:
        return render_template('4th_test.html',img1 = img1, img2=img2, img3=img3, img4=img4)
    
    




################### 5번째 게임 : 파이게임(기억력) ###################
@app.route('/pygame')
def pygame():
    return render_template('5th_test.html')


@app.route('/get_screenshot', methods=['POST'])
def get_screenshot():
    
    # 기억력 게임 점수에 대한 함수 정의 
    def get_score(level) :
        if level == 1:
            score = 0
        elif level == 2:
            score = 1
        elif 3 <= level <= 4:
            score = 2
        elif level == 5:
            score = 3
        else:
            score = 4
            
        return level, score

    # 기억력 게임을 완료한 이후 easyocr을 이용해 게임결과 이미지에서 텍스트추출
    im = pyscreenshot.grab()
    random_id = ''.join([random.choice(string.ascii_letters + string.digits) for n in range(32)])
    file_name = 'static/5/img/{}.png'.format(random_id)
    im.save(file_name)
    reader = easyocr.Reader(['ko', 'en'])
    
    with open(file_name,'rb') as pf:
        img = pf.read()
        result = reader.readtext(img)
        for res in result:
            if res[1][0:10] == 'Your level':    
                level = res[1][-1]
                result = get_score(int(level))
    
    # 텍스트로 추출한 결과를 DB에 저장
    conn = sql.connect('remember.db', isolation_level=None)
    cur = conn.cursor()
    cur.execute(
        'CREATE TABLE IF NOT EXISTS remember (level TEXT, score TEXT)')
    cur.execute("""INSERT INTO remember(level, score) 
                    VALUES(?, ?)""", (result[0], result[1]))
    conn.commit()
    cur.close()
                
    os.remove(file_name)



DATABASE_URI = 'sttdb.db'
# ---- DB에서 데이터를 불러오기 ----
conn = sql.connect(DATABASE_URI, isolation_level=None)
cur = conn.cursor()
# cur.execute(
#     'CREATE TABLE IF NOT EXISTS STT (id TEXT, p TEXT, url TEXT)')
# id = 1
# p = "안녕하세요. 오늘도 멋진 하루 되세요"
# url = 'C:/Users/admin/Desktop/SoftLand (1)/game/6_stt/sound/정답1.wav'

# conn = sql.connect(DATABASE_URI, isolation_level=None)
# cur = conn.cursor()
# cur.execute("""INSERT INTO STT(id, p, url) 
#                     VALUES(?, ?, ?)""", (id, p, url))

cur.execute("SELECT * FROM STT")
db_text = str(cur.fetchmany(size=1))
print(db_text)

# 경로와 정답Text만 추출하기 위한 처리
db_List = db_text.split("'")

global sound_target
sound_url = db_List[5]    # 경로
sound_target = db_List[3] # 정답Text
dic = {'1' : sound_target} # 정답 Text


@app.route('/sound')
def sound():
    
    return render_template('6th_test.html', target=dic['1'])

@app.route('/STT', methods=['POST', 'GET'])
def STT():
    
    String_sound = ''  # 녹음파일 Text
    String_target = '' # 정답 Text
    
    sleep(5)
    count = 1
    
    #---------------------------------------------------------------------------
    #      STT Open API
    #---------------------------------------------------------------------------
    if request.method == 'POST':
        openApiURL = "http://aiopen.etri.re.kr:8000/WiseASR/Recognition"
        accessKey = "f0f9fd15-daef-4655-b516-d7a9711c696a" 
        audioFilePath = "C:/Users/admin/Downloads/정답1.wav" # 다운로드한 음성파일을 여기에 넣어서 Text로 바꾸기
            
        languageCode = "korean"
        
        file = open(audioFilePath, "rb")
        audioContents = base64.b64encode(file.read()).decode("utf8")
        file.close()
        
        requestJson = {    
            "argument": {
                "language_code": languageCode,
                "audio": audioContents
            }
        }
        
        http = urllib3.PoolManager()
        response = http.request(
        "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8","Authorization": accessKey},
            body=json.dumps(requestJson)
        )
        
        print("[responseCode] " + str(response.status))
        print("[responBody]")
        print("===== 결과 확인 ====")

        # 출력결과는 쓸때없는 내용이 들어가기 때문에 필요한 부분만 가져오기
        string = str(response.data,"utf-8")
        List = string.split('"')
        List = List[-2]
        List = List[:-1]
        print(List)
        # 녹음한 음성을 처리한 결과를 List변수에 담는다.
        
        
        # dic = {'1' : "안녕하세요. 오늘도 멋진 하루 되세요"}
        
        
        # NLP 유사도검사를 위해 정답Text와 녹음하고 Text로 바꾼 결과를 변수에 담에서 NLP모델에 넘긴다.
        # 녹음파일 Text
        String_sound = List
        
        # 정답Text
        String_target = sound_target
        
        print(List)
        
        #---------------------------------------------------------------------------
        #       유사도 검사 NLP Open API
        #---------------------------------------------------------------------------
        
        openApiURL = "http://aiopen.etri.re.kr:8000/ParaphraseQA"
        accessKey = "f0f9fd15-daef-4655-b516-d7a9711c696a"
        sentence1 = String_sound
        sentence2 = String_target
        
        requestJson = {
        "argument": {
            "sentence1": sentence1,
            "sentence2": sentence2
            }
        }
        
        http = urllib3.PoolManager()
        response = http.request(
            "POST",
            openApiURL,
            headers={"Content-Type": "application/json; charset=UTF-8","Authorization" :  accessKey},
            body=json.dumps(requestJson)
        )
        
        print("[responseCode] " + str(response.status))
        print("[responBody]")
        print(str(response.data,"utf-8"))

        NLP_String = str(response.data,"utf-8")
        NLP_List = NLP_String.split('"')
        print(NLP_List)
        
        NLP_reuslt = NLP_List[-2]
        # NLP_reuslt = NLP_target[:-1]
        print(NLP_reuslt)
        
        #--------------------------------------------------------------------------
        #     검증 결과 추출 및 전송
        #--------------------------------------------------------------------------
        
        String = ''
        if NLP_reuslt == 'paraphrase' :
            String += '유사합니다'
        else:
            String += '유사하지 않습니다'
            
        os.remove(audioFilePath)
        #                                             정답문장          TTS        체크 결과
        return render_template('6th_test.html', target = sentence2, sound = sentence1, ck=String)



################### 6번째 게임 : STT ###################
@app.route('/stt')
def stt():
    return render_template('6th_test.html')

# 6st test



################### 결과페이지 : 대시보드 ###################
@app.route('/result')
def result():
    return render_template('result_2.html')


@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')

@app.route('/abouttest')
def abouttest():
    return render_template('abouttest.html')




if __name__ == '__main__':
    # https://flask.palletsprojects.com/en/2.0.x/api/#flask.Flask.run
    # https://snacky.tistory.com/9
     # host주소와 port number 선언
    app.run(host='0.0.0.0', debug=True)  
    
    
    
    