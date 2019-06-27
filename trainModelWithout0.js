const tf = require("@tensorflow/tfjs")
const fs = require('fs')
const request = require('request')
require('@tensorflow/tfjs-node')

var data = [];
var MAX = -999;
var sequenceLen = 10
var trainData = []
var labelData = []
var validateData = []
var validateLabel = []
var validateSize = 200

function getData() {
    return new Promise(function(resolve, reject) {
        request('http://202.139.192.82/sanam/8000', { json: true }, (err, res, body) => {
            if (err) {
                return console.log(err)
            }
            body.number_of_tourist.reverse().forEach(function(item) {
                if (item != 0) {
                    data.push(item)
                }
            })
            resolve(data)
        })
    })
}

async function prepareData() {
    MAX = -999;
    const len = data.length
    for (i=0; i<len; i++) {
        if (MAX <= data[i]) {
            MAX = data[i];
        }
    }
    
    //let dataset = data;
    let dataset = data.map((number) => {
        return number/MAX;
    })
    var i, j
    var temp = []
    for(i = 0; i < dataset.length - sequenceLen - 2; i++) {
    	temp = []
    	for(j = 0; j < sequenceLen; j++) {
    		temp.push(dataset[j+i])
    	}
        trainData.push(temp)
        temp = []
        for(j = 0; j < 3; j++) {
            temp.push(dataset[i + sequenceLen + j])
        }
    	labelData.push(temp)
    }
    // validateData = trainData.slice(trainData.length - validateSize, trainData.length)
    // validateLabel = labelData.slice(labelData.length - validateSize, labelData.length)
    // trainData = trainData.slice(0, labelData.length - validateSize)
    // labelData = labelData.slice(0, labelData.length - validateSize)
}

const model = tf.sequential();

model.add(tf.layers.lstm({
    units: 100,
    inputShape: [sequenceLen, 1],
    returnSequences: false
}));

model.add(tf.layers.dense({
    units: 3,
    kernelInitializer: 'VarianceScaling',
    activation: 'relu'
}));

const LEARNING_RATE = 0.3;
const optimizer = tf.train.adam();

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
                batchSize: 5,
                epochs: 18,
                shuffle: true,
                validationSplit: 0.2
            });
    }

    await getData()
    await prepareData()

    //console.log(labelData)
    //console.log(trainData)
    var trainXS = tf.reshape(tf.tensor2d(trainData), [-1,sequenceLen,1])
    var trainYS = tf.reshape(tf.tensor2d(labelData), [-1,3])

    await trainModel()

    const saveResult = await model.save('file://model1');

    const r = model.predict(tf.reshape(tf.tensor2d(trainData), [-1,sequenceLen,1]));
    let trainDataPredict = r.dataSync();
    // const s = model.predict(tf.reshape(tf.tensor2d(validateData), [-1,sequenceLen,1]));
    // let validateDataPredict = s.dataSync();
    
    let temp1 = []
    let temp2 = []
    let temp3 = []
    let i = 0
    for(i = 0 ; i < trainDataPredict.length; i+=3) {
        temp1.push(trainDataPredict[i]*MAX)
        temp2.push(trainDataPredict[i+1]*MAX)
        temp3.push(trainDataPredict[i+2]*MAX)
    }
    console.log(temp1.length)
    console.log(temp2.length)
    console.log(temp3.length)
	fs.writeFileSync('temp1.json', JSON.stringify(temp1));
    fs.writeFileSync('temp2.json', JSON.stringify(temp2));
    fs.writeFileSync('temp3.json', JSON.stringify(temp3));
    // for(i = 0 ; i < validateData.length; i++) {
    // 	temp.push(validateDataPredict[i]*MAX)
    // }
    // fs.writeFileSync('validateDataPredict.json', JSON.stringify(temp)); 
}

main();