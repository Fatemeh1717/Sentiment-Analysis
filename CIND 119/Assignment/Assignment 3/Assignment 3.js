#1
use Sample

db.createCollection("test_data")
#2
db.test_data.insertMany([{
  
  "Class": "No",
  "age": 35,
  "menopause": "premeno",
  "deg_malig": 3,
  "breast": "left",
  "breast_quad": "left_low",
  "irradiat": "no"
},{
  "Class": "No",
  "age": 42,
  "menopause": "premeno",
  "deg_malig": 2,
  "breast": "right",
  "breast_quad": "right_up",
  "irradiat": "no"
},{
  "Class": "No",
  "age": 30,
  "menopause": "premeno",
  "deg_malig": 2,
  "breast": "left",
  "breast_quad": "left_low",
  "irradiat": "no"
},{
  "Class": "No",
  "age": 61,
  "menopause": "ge40",
  "deg_malig": 2,
  "breast": "right",
  "breast_quad": "left_up",
  "irradiat": "no"
},{
  "Class": "No",
  "age": 45,
  "menopause": "premeno",
  "deg_malig": 2,
  "breast": "right",
  "breast_quad": "right_low",
  "irradiat": "no"
},{
  "Class": "No",
  "age": 64,
  "menopause": "ge40",
  "deg_malig": 2,
  "breast": "left",
  "breast_quad": "left_low",
  "irradiat": "no"
},{
  "Class": "No",
  "age": 52,
  "menopause": "premeno",
  "deg_malig": 2,
  "breast": "left",
  "breast_quad": "left_low",
  "irradiat": "no"
},{
  "Class": "No",
  "age": 67,
  "menopause": "ge40",
  "deg_malig": 1,
  "breast": "left",
  "breast_quad": "left_low",
  "irradiat": "no"
},{
  "Class": "Yes",
  "age": 41,
  "menopause": "premeno",
  "deg_malig": 2,
  "breast": "left",
  "breast_quad": "left_low",
  "irradiat": "no"
},{
  "Class": "Yes",
  "age": 43,
  "menopause": "premeno",
  "deg_malig": 2,
  "breast": "right",
  "breast_quad": "left_up",
  "irradiat": "no"
},{
  "Class": "Yes",
  "age": 41,
  "menopause": "premeno",
  "deg_malig": 3,
  "breast": "left",
  "breast_quad": "central",
  "irradiat": "no"
},{
  "Class": "Yes",
  "age": 44,
  "menopause": "ge40",
  "deg_malig": 2,
  "breast": "left",
  "breast_quad": "left_low",
  "irradiat": "no"
},{
  "Class": "Yes",
  "age": 61,
  "menopause": "It40",
  "deg_malig": 1,
  "breast": "left",
  "breast_quad": "right_up",
  "irradiat": "no"
},{
  "Class": "Yes",
  "age": 55,
  "menopause": "ge40",
  "deg_malig": 3,
  "breast": "left",
  "breast_quad": "right_up",
  "irradiat": "no"
},{
  "Class": "Yes",
  "age": 44,
  "menopause": "premeno",
  "deg_malig": 3,
  "breast": "left",
  "breast_quad": "left_up",
  "irradiat": "no"
}])
# 3.a
db.test_data.find({
    
    menopause:"ge40"
    
    })
    
# 3.b

db.test_data.find({
    
    age:{$lt : 41}
    
    })
    
# 3.c
    
db.test_data.find({
    
    $or:[{age:{$lt:41}},{menopause:"ge40"}]
    
     
    })

# 3.d
    
db.test_data.aggregate( [{$group:{_id:null, avg_age:{$avg:"$age"}}}])