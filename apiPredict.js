var bodyParser = require('body-parser');
var express = require('express');
var cors = require('cors');  
var app = express();
var tf = require("@tensorflow/tfjs")
const request = require('request')
require('@tensorflow/tfjs-node')
port = process.env.PORT || 12501;

app.use(cors());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());
app.listen(port);

const sequenceLen = 10
const normalize_value = 1800
var model
async function loadModelTF() {
	console.log("model is loading")
	model = await tf.loadModel('file://model1-best/model.json');
	console.log("model loaded")
	console.log('API server started on: ' + port); 
}

loadModelTF()

app.get("/testml", function(req, res) {
	request('http://202.139.192.82/ml/last-hours?hours=18', { json: true }, (err, ress, body) => {
        if (err) {
            res("")
        }
        let data = []
        body.number_of_tourist.reverse().forEach(function(item) {
        	data.push(item/normalize_value)
        })
        data = data.slice(0,10)
        console.log(data)
        let predictData = tf.reshape(tf.tensor2d([data]), [-1,sequenceLen,1])
        const r = model.predict(predictData);
		let predictResult = r.dataSync();
		let result = []
		let i = 0
		for (i = 0; i < predictResult.length; i++) {
			result.push(Math.round(predictResult[i] * normalize_value)) 
		}

		res.json({"number_of_tourist":result})
    })	
})

app.post("/testml", function(req, res) {
    let data = req.body.list
    let predictData = tf.reshape(tf.tensor2d([data]), [-1,sequenceLen,1])
    const r = model.predict(predictData);
	let predictResult = r.dataSync();
	let result = []
	let i = 0
	for (i = 0; i < predictResult.length; i++) {
		result.push(Math.round(predictResult[i] * normalize_value)) 
	}
	res.json({"number_of_tourist":result})
})	
		


