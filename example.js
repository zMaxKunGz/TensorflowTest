const tf = require('@tensorflow/tfjs');
require('@tensorflow/tfjs-node');
var csv = require("fast-csv");

var xs = [];
var ys = [];

function range(start, end) {
    var ans = [];
    for (let i = start; i < end; i++) {
        ans.push(i);
    }
    return ans;
}

function readCSV() {
    return new Promise(function(resolve, reject) {
        csv
        .fromPath(file_name) 
        .on("data", function(str){    
        })
        .on("end", function(){
        
        console.log(data.length);
        resolve(data);
        });
    });
}


async function prepareData() {
    /*
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

    let arr = range(TIME_STEP, dataset.length - NUM_OUT + 1);

    arr.forEach(function(i) {
        
    });
    */
   xs = [[10,20,30],[20,30,40],[30,40,50],[10,20,30],[20,30,40]];
   ys = [40,50,60,40,50];
    
}

const model = tf.sequential();

model.add(tf.layers.lstm({
    units: 50,
    inputShape: [3, 1],
    returnSequences: false
}));

model.add(tf.layers.dense({
    units: 1,
    kernelInitializer: 'VarianceScaling',
    activation: 'relu'
}));

const LEARNING_RATE = 0.2;
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
                epochs: 100,
                shuffle: true,
                validationSplit: 0.2
            });
    }
    await prepareData();
    var trainXS = tf.tensor2d(xs)
    trainXS = tf.reshape(trainXS, [-1, 3, 1])
    //trainXS.print()

    var trainYS = tf.tensor1d(ys)
    trainYS = tf.reshape(trainYS, [-1, 1])
    
	await trainModel();

    const saveResult = await model.save('file://mymodel');
    
    const load = async () => {
        const model = await tf.loadModel('file://mymodel/model.json');
      };
      
    await load();

    const r = model.predict(tf.reshape(tf.tensor2d([[30,40,50]]), [-1,3,1]));
    let result = r.dataSync()[0];
    console.log(result);
}

main();