function verifyTruthTable(network, expectedResult, operation) {
    var ctr = 0;
    for(var j=0;j<=1;j++) {
        for(var k=0;k<=1;k++) {
            var score = network.input([j,k]).predict();
            var prediction = score > 0.5 ? true : false;
            var expected = expectedResult[ctr++];
            var res = prediction == expected ? 'PASS! ' : 'FAIL! ';

            console.log(res + j + ' ' + operation + ' ' + k + ' = ' +
                (expected ? true : false) + ' (prediction: ' + prediction + ' ( score: ' + score + '))');
        }
    }
}

function generateLogicNetwork() {
    var input = new ThinqNeuronLayer('input',2);
    var hidden = new ThinqNeuronLayer('hidden',2, 'sigmoid');
    var output = new ThinqNeuronLayer('output',1, 'sigmoid');
    return new ThinqNeuralNet(input, hidden, output);
}
