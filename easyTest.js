const tf = require("@tensorflow/tfjs")
require('@tensorflow/tfjs-node')
const fs = require("fs")
const csvtojson = require("csvtojson");
// convert/setup our data

function shuffleArray(array) {
    for (var i = array.length - 1; i > 0; i--) {
        var j = Math.floor(Math.random() * (i + 1));
        var temp = array[i];
        array[i] = array[j];
        array[j] = temp;
    }
}

csvtojson().fromFile("SalaryData.csv").then((jsonObj)=> {
	shuffleArray(jsonObj)
    let data = jsonObj.map(item => [Number(item.YearsExperience)])
    let label = jsonObj.map(item => [Number(item.Salary)])

   	trainingData = data.slice(10)
    trainingLabel = label.slice(10)
    verifyData = data.slice(0, 10)
    verifyLabel = label.slice(0, 10)

    trainingData = tf.tensor2d(trainingData)
    trainingLabel = tf.tensor2d(trainingLabel)
    verifyData = tf.tensor2d(verifyData)
    verifyLabel = tf.tensor2d(verifyLabel)

    verifyData.print()

    const model = tf.sequential()

	model.add(tf.layers.dense({
		inputShape: [1],
		activation: "relu",
		units: 2,
	}))
	model.add(tf.layers.dense({
		activation: "relu",
		units: 2,
	}))
	model.add(tf.layers.dense({
		activation: "sigmoid",
		units: 1,
	}))
	model.compile({
		loss: "meanSquaredError",
		optimizer: tf.train.adam(.05),
	})

	model.fit(trainingData, trainingLabel, {epochs: 100})
	  .then((history) => {
	    model.predict(verifyData).print()
	  })
  })


// const trainingData = tf.tensor2d(iris.map(item => [
//   item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
// ]))

// const outputData = tf.tensor2d(iris.map(item => [
//   item.species === "setosa" ? 1 : 0,
//   item.species === "virginica" ? 1 : 0,
//   item.species === "versicolor" ? 1 : 0,
// ]))
// const testingData = tf.tensor2d(irisTesting.map(item => [
//   item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
// ]))

// build neural network
