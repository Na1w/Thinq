// Basic verification method

var xorNet = [{"n":[],"s":2,"a":"","p":"input"},{"n":[2.993349075317383,1.2076431512832642,-1.7963570356369019,2.4834094047546387,2.2369771003723145,-1.5567731857299805,1,0.6202237010002136],"s":3,"a":"tanh","p":"hidden"},{"n":[5.638411045074463,-3.966381549835205,-4.385454177856445,1,-1.5523728132247925],"s":1,"a":"sigmoid","p":"output"}];
var andNet = [{"n":[],"s":2,"a":"","p":"input"},{"n":[1.4060430526733398,1.4140589237213135,1.6646751165390015,1.5921272039413452,1.5354925394058228,1.615476131439209,1,-2.1499767303466797],"s":3,"a":"tanh","p":"hidden"},{"n":[1.405582070350647,2.7589821815490723,2.389249563217163,1,-0.8945946097373962],"s":1,"a":"sigmoid","p":"output"}];
var orNet = [{"n":[],"s":2,"a":"","p":"input"},{"n":[1.5154386758804321,1.9394458532333374,1.4864941835403442,1.4138015508651733,1.8241556882858276,1.5214253664016724,1,-0.8154666423797607],"s":3,"a":"tanh","p":"hidden"},{"n":[2.2753586769104004,1.7675362825393677,2.2783865928649902,1,0.24834518134593964],"s":1,"a":"sigmoid","p":"output"}];
var norNet = [{"n":[],"s":2,"a":"","p":"input"},{"n":[1.4766144752502441,1.5731358528137207,1.8701391220092773,1.9548149108886719,1.4094535112380981,1.260497808456421,1,-0.823724091053009],"s":3,"a":"tanh","p":"hidden"},{"n":[-1.9985570907592773,-2.582667827606201,-1.7641922235488892,1,-0.311291366815567],"s":1,"a":"sigmoid","p":"output"}];

var xorANN = new ThinqNeuralNet().loadANN(xorNet);
var andANN = new ThinqNeuralNet().loadANN(andNet);
var orANN = new ThinqNeuralNet().loadANN(orNet);
var norANN = new ThinqNeuralNet().loadANN(norNet);

verifyTruthTable(orANN, [0,1,1,1], '||');
verifyTruthTable(andANN, [0,0,0,1], '&&');
verifyTruthTable(xorANN, [0,1,1,0], '^');
verifyTruthTable(norANN, [1,0,0,0], '~||');



console.log('*****************************************************************************');

// Example Training
var ann = new ThinqNeuralNet(new ThinqNeuronLayer('input',2),
    new ThinqNeuronLayer('hidden', 3, 'tanh'),
    new ThinqNeuronLayer('output', 1, 'sigmoid'));

// Eftersom nätverken initialiseras med slumpvärden så är det inte alltid
// träningen ger meningsfulla resultat- utan kan behöva köras om
//
// Det är så det fungerar... ingen exakt vetenskap.

console.log('Training sample XOR net',ann);
var j = 0;
for(var i=0;i<500;i++)
{
    ann.fit([0,0],[0],1);
    ann.fit([0,1],[1],1);
    ann.fit([1,0],[1],1);
    ann.fit([1,1],[0],1);
}

console.log('Total train time ' + (ann.totalTime/1000) + ' seconds');
console.log('Testing trained network')
verifyTruthTable(ann, [0,1,1,0], 'Xor')
console.log('Export: ', JSON.stringify(ann.saveANN()));
