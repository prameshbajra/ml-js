// const model = tf.sequential();

// const hiddenLayer = tf.layers.dense({
//     units: 4,
//     inputShape: [2],
//     activation: 'sigmoid'
// });

// model.add(hiddenLayer);

// const outputLayer = tf.layers.dense({
//     units: 1,
//     activation: 'sigmoid'
// })

// model.add(outputLayer);

// const sOpt = tf.train.sgd(0.5);   // Rate of learing ... sgd = sarcastic gradient descent

// model.compile({
//     optimizer: sOpt,
//     loss: 'meanSquaredError'
// });

// const xs = tf.tensor2d([
//     [0, 0],
//     [0.5, 0.5],
//     [1, 1]
// ]);

// const ys = tf.tensor2d([
//     [1],
//     [0.5],
//     [0]
// ]);

// async function fitter() {
//     for (i = 0; i < 100; i++) {
//         const config = {
//             epochs: 10,
//             shuffle: true
//         };
//         const response = await model.fit(xs, ys, config);
//         console.log(response.history.loss[0])
//     }
// }

// fitter().then(() => {
//     const finalResult = model.predict(xs);
//     finalResult.print();
// })

const model = tf.sequential();

// Hidden layer ...
model.add(tf.layers.dense({
    units: 3,
    inputShape: [2],
    activaton: 'sigmoid'
}));

// OUtput layer ...
model.add(tf.layers.dense({
    units: 2,
    activaton: 'sigmoid'
}));

const xs = tf.tensor2d([
    [0.4, 0.2],
    [0.9, 0.5]
]);

const ys = tf.tensor2d([
    [0.8, 0.6],
    [0.1, 0.5]
])

const sgd = tf.train.sgd(0.5);

model.compile({
    optimizer: sgd,
    loss: 'meanSquaredError'
})

async function fitter() {
    for (i = 0; i < 100; i++) {
        const res = await model.fit(xs, ys, {
            epochs: 10,
            shuffle: true
        });
        console.log(res.history.loss[0]);
    }
}

fitter().then(() => {
    const finalResult = model.predict(xs);
    finalResult.print();
})