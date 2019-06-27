const tf = require("@tensorflow/tfjs")
require('@tensorflow/tfjs-node')
const csvtojson = require("csvtojson");
const csv = require("fast-csv");
const fs = require('fs');

var file_name = "THB.csv";
var data = [];
var MAX = -999;
var sequenceLen = 6
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
	        //console.log(data.length);
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
    // console.log(trainData.length)
    validateData = trainData.slice(trainData.length - validateSize, trainData.length)
    validateLabel = labelData.slice(labelData.length - validateSize, labelData.length)
    trainData = trainData.slice(0, labelData.length - validateSize)
    labelData = labelData.slice(0, labelData.length - validateSize)
}

const model = tf.sequential();

model.add(tf.layers.lstm({
    units: 20,
    inputShape: [sequenceLen, 1],
    returnSequences: false
}));

model.add(tf.layers.dense({
    units: 1,
    kernelInitializer: 'VarianceScaling',
    activation: 'relu'
}));

const LEARNING_RATE = 0.2;
const optimizer = tf.train.sgd(LEARNING_RATE);

model.compile({
    optimizer: optimizer,
    loss: 'meanSquaredError',
    metrics: ['accuracy'],
});

async function main(){
	async function trainModel(){
        const history = await model.fit(
            trainXS,
            trainYS,
            {
                batchSize: 50,
                epochs: 2,
                shuffle: true,
                validationSplit: 0.3
            });
    }

    var data = await readCSV();

    await prepareData()
    
    console.log()

    var trainXS = tf.tensor2d(trainData)
    trainXS = tf.reshape(trainXS, [-1, sequenceLen, 1])

    var trainYS = tf.tensor1d(labelData)
    trainYS = tf.reshape(trainYS, [-1, 1])
    
	await trainModel();

    const saveResult = await model.save('file://mymodel');
    
    const load = async () => {
        const model = await tf.loadModel('file://exported_model/model.json');
    };
      
    await load();

    const r = model.predict(tf.reshape(tf.tensor2d(trainData), [-1,sequenceLen,1]));
    let trainDataPredict = r.dataSync();
    const s = model.predict(tf.reshape(tf.tensor2d(validateData), [-1,sequenceLen,1]));
    let validateDataPredict = s.dataSync();

    let temp = []
    let i = 0
    for(i = 0 ; i < trainData.length; i++) {
    	temp.push(trainDataPredict[i])
    }
	fs.writeFileSync('trainDataPredict.json', JSON.stringify(temp)); 
    temp = []
    for(i = 0 ; i < validateData.length; i++) {
    	temp.push(validateDataPredict[i])
    }
    fs.writeFileSync('validateDataPredict.json', JSON.stringify(temp)); 
}

main();