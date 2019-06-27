const tf = require("@tensorflow/tfjs")
require('@tensorflow/tfjs-node')
const csvtojson = require("csvtojson");
const csv = require("fast-csv");
const fs = require('fs');

var file_name = "THB.csv";
var data = [];
var MAX = -999;
var sequenceLen = 10
var trainData = []
var labelData = []
var validateData = []
var validateLabel = []
var validateSize = 200

function readCSV() {
    return new Promise(function(resolve, reject) {
        csv
        .fromPath(file_name) 
        .on("data", function(str) {   
        	data.push(Number(str[1]))
        })
        .on("end", function(){
	        resolve(data);
        });
    });
}

async function prepareData() {
    MAX = -999;
    const len = data.length
    for (i=0; i<len; i++) {
        if (MAX <= data[i]) {
            MAX = data[i];
        }
    }
    
    let dataset = data.map((number) => {
        return number/MAX;
    })
    var i, j
    var temp = []
    for(i = 0; i < dataset.length - sequenceLen; i++) {
    	temp = []
    	for(j = 0; j < sequenceLen; j++) {
    		temp.push(dataset[j+i])
    	}
    	trainData.push(temp)
    	labelData.push(dataset[i+j])
    }
    validateData = trainData.slice(trainData.length - validateSize, trainData.length)
    validateLabel = labelData.slice(labelData.length - validateSize, labelData.length)
    // trainData = trainData.slice(0, labelData.length - validateSize)
    // labelData = labelData.slice(0, labelData.length - validateSize)
}

async function main(){

    var data = await readCSV();

    await prepareData()
    
    const model = await tf.loadModel('file://exported_modelNew/model.json');

    const r = model.predict(tf.reshape(tf.tensor2d(trainData), [-1,sequenceLen,1]));
    let trainDataPredict = r.dataSync();
    // const s = model.predict(tf.reshape(tf.tensor2d(validateData), [-1,sequenceLen,1]));
    // let validateDataPredict = s.dataSync();
    
    let temp = []
    let i = 0
    for(i = 0 ; i < trainData.length; i++) {
    	temp.push(trainDataPredict[i]*MAX)
    }
	fs.writeFileSync('trainDataPredict.json', JSON.stringify(temp)); 
    // temp = []
    // for(i = 0 ; i < validateData.length; i++) {
    // 	temp.push(validateDataPredict[i]*MAX)
    // }
    // fs.writeFileSync('validateDataPredict.json', JSON.stringify(temp)); 
}

main();