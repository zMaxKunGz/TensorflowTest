const tf = require("@tensorflow/tfjs")
const fs = require('fs')
const request = require('request')
require('@tensorflow/tfjs-node')
const dataJson = require('./newData.json')

var data = [];
var MAX = -999;
var sequenceLen = 7
var trainData = []
var labelData = []
var validateData = []
var validateLabel = []
var validateSize = 200
var newData = []

async function getData() {
        // request('http://202.139.192.82/ml/last-hours?hours=8000', { json: true }, (err, res, body) => {
        //     if (err) {
        //         return console.log(err)
        //     }
        //     let i = 0
        //     let temp = []
        //     for (i = 0; i < body.number_of_tourist.length; i++) {
        //         if (body.number_of_tourist[i] != 0) {

        //         
        //             temp.push(body.number_of_tourist[i])
        //             if (i != 0 && i % 24 == 0) {
        //                 data.push(temp);
            //             temp = []
            //         }
       // }
        //     }
        //     data.push(temp)
        //     resolve(data)
        // })
    
    let i = 0
    let temp = []
    let body = dataJson.number_of_tourist
    for (i = 0; i < body.length; i++) {
        if (MAX <= body[i]) {
            MAX = body[i];
        }
        if (body[i] != 0) {
            temp.push(body[i])
        }
        if (i != 0 && i % 24 == 0) {
            data.push(temp);
            temp = []
        }
    }
    //data.push(temp)
}

async function prepareData() {
    //let dataset = data;
    var i, j
    let temp = []
    data.forEach(function (item) {
        item = item.reverse()
        for(i = 0; i < item.length - sequenceLen - 2; i++) {
            temp = []
            for(j = 0; j < sequenceLen; j++) {
                temp.push(item[j+i]/MAX)
            }
            trainData.push(temp)
            temp = []
            for(j = 0; j < 3; j++) {
                temp.push(item[i + sequenceLen + j]/MAX)
            }
            labelData.push(temp)
        }
    })
    
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
const optimizer = tf.train.adam(LEARNING_RATE);

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
                batchSize: 1,
                epochs: 2,
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

    const saveResult = await model.save('file://model2');

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
    fs.writeFileSync('data.json', JSON.stringify(newData));
	fs.writeFileSync('temp1.json', JSON.stringify(temp1));
    fs.writeFileSync('temp2.json', JSON.stringify(temp2));
    fs.writeFileSync('temp3.json', JSON.stringify(temp3));
    
    for(i = 0 ; i < validateData.length; i++) {
    	temp.push(validateDataPredict[i]*MAX)
    }
    fs.writeFileSync('validateDataPredict.json', JSON.stringify(temp)); 
}

main();