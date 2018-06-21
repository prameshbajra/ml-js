let xs = [];
let ys = [];
let learningRate = 0.24
let optimizer = tf.train.sgd(learningRate);

function setup() {
    createCanvas(500, 500);
    m = tf.variable(tf.scalar(random(1)));
    b = tf.variable(tf.scalar(random(1)));
}

function predict(xs) {
    const tfxs = tf.tensor1d(xs);
    const ys = tfxs.mul(m).add(b);
    return ys;
}

function lossFunction(predictedYs, realYs) {
    return predictedYs.sub(realYs).square().mean();
}

function mousePressed() {
    xs.push(map(mouseX, 0, width, 0, 1));
    ys.push(map(mouseY, 0, height, 1, 0));
}
function draw() {
    tf.tidy(() => {
        if (ys.length > 0) {
            const tfys = tf.tensor1d(ys);
            optimizer.minimize(() => lossFunction(predict(xs), tfys));
        }
    })
    background(0);
    stroke(255);
    strokeWeight(8);
    xs.forEach((element, i) => { point(map(element, 0, 1, 0, width), map(ys[i], 0, 1, height, 0)) });

    const xCords = [0, 1];
    const tfyCords = tf.tidy(() => predict(xCords));  /// This will generate tensors so , we nee the bumbers back ...
    const yCords = tfyCords.dataSync();
    tfyCords.dispose();
    const x1 = map(xCords[0], 0, 1, 0, width);
    const x2 = map(xCords[1], 0, 1, 0, width);
    const y1 = map(yCords[0], 0, 1, height, 0);
    const y2 = map(yCords[1], 0, 1, height, 0);
    line(x1, y1, x2, y2)
    console.log(tf.memory().numTensors);
}