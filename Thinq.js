/*
    Thinq 0.2


    Fredrik Andersson -2019
    <fredrikandersson@mac.com>
*/

class ThinqNeuronLayer {
    constructor(name, size, activator) {
        this.name = name;
        this.size = size;
        this.value = new Float32Array(size).fill(0);
        this.weights = null;
        this.seed = null;
        this.activation = activator;

        switch(activator) {
            case 'relu':
                this.activator = function (x) { return Math.max(0,x); };
                this.derivative = function (x) { return x > 0 ? 1 : 0; };
                this.seed = function () { return Math.random()+.01; };
                break;
            case 'tanh':
                this.activator = function (x) { return Math.tanh(x); };
                this.derivative = function (x) { return 1-Math.pow(Math.tanh(x), 2); };
                this.seed = function () { return Math.random()*2-1; };
                break;
            case 'softmax':
                this.activator = function (ignore, no) {
                    var sum=0;
                    for(var i=0;i<this.size;i++) {
                        sum+=Math.exp(this.value[i]);
                    }
                    return Math.exp(this.value[no]) / sum;
                };
                this.derivative = function () { return 0;};
                this.seed = function () { return 0;};

                break;
            default: // sigmoid
                this.activation = 'sigmoid';
                this.activator = function (x) { return 1/(1+Math.exp(-x));};
                this.derivative = function (x) { return x * (1 - x); };
                this.seed = function () { var x = Math.random()*2; return Math.sqrt(x); };
                break;
        }
    }

    wire(target, restoreState) {
        this.targetSize = target.getSize();
        if(!restoreState) {
            target.weights = new Float32Array(this.size * this.targetSize + 2);
            for(var p=0;p<this.size;p++) {
                for(var c=0;c<this.targetSize;c++) {
                    target.weights[c*this.size + p] = this.seed();
                }
            }
            target.weights[target.weights.length-2] = 1;
            target.weights[target.weights.length-1] = 0;
        } else {
            target.restoreState(restoreState);
        }

        return target;
    }

    restoreState(state) {
        this.weights = Float32Array.from(state);
        return this;
    }

    getRestoreState() {
        return Array.from(this.weights||[]);
    }

    project(values) {
        this.value = Float32Array.from(values);
        return this;
    }

    propagate(input_net, in_errors, learning_rate) {
        var errors = new Float32Array(input_net.getSize());
        var deltas = new Float32Array(this.getSize()).fill(0);

        var row_distance = input_net.getSize();
        for(var cell=0;cell<this.getSize();cell++) {
            var derivative = this.derivative(this.value[cell]);
            deltas[cell] = derivative * in_errors[cell];
        }

        for(var sW=0;sW<input_net.getSize();sW++) {
            errors[sW] = 0;
            for(var n=0;n<this.getSize();n++) {
                errors[sW] += this.weights[n * row_distance + sW] * deltas[n];
            }
        }

        for(var n=0;n<this.getSize();n++) {
            for(var i=0;i<input_net.getSize();i++) {
                var weight_delta = learning_rate * input_net.value[i] * deltas[n];
                this.weights[n * row_distance + i] -= weight_delta;
            }
            this.weights[this.weights.length-1] -= learning_rate * 1 * deltas[n];
        }

        return errors;
    }

    compute(in_net) {
        for(var cell=0;cell<this.getSize();cell++) {
            var cellValue = 0;
            for(var sV=0;sV<in_net.getSize();sV++) {
                cellValue += in_net.value[sV] * this.weights[cell * in_net.getSize() + sV];
            }
            cellValue += this.weights[this.weights.length-2] * this.weights[this.weights.length-1];
            this.value[cell] = this.activator(cellValue, cell);
        }
        return this;
    }

    getValues() {
        return this.value;
    }

    getSize() {
        return this.size;
    }
}

class ThinqNeuralNet{
    constructor(va_args)
    {
        this.totalTime = 0;
        if(!va_args)
            return;
        if(arguments.length < 2) {
            throw new Error("Must have at least an input and an output layer!");
        }
        this.network = [];
        for(var i=0;i<arguments.length;i++) {
            this.network.push(arguments[i]);
        }
        this.inputs = new Float32Array(this.network[0].getSize());
        for(var i=0;i<this.network.length-1;i++) {
            this.network[i].wire(this.network[i+1]);
        }
    }

    input(f64array) {
        this.inputs = Float32Array.from(f64array);
        return this;
    }

    loadANN(saveState) {
        this.network = [];
        for(let net of saveState) {
            var layer = new ThinqNeuronLayer(net.p, net.s, net.a);
            layer.restoreState(net.n);
            this.network.push(layer);
        }
        this.inputs = new Float32Array(this.network[0].getSize());
        for(var i=0;i<this.network.length-1;i++) {
            this.network[i].wire(this.network[i+1], this.network[i+1].weights);
        }
        return this;
    }

    saveANN() {
        var sum = [];
        for(var i=0;i<this.network.length;i++) {
            sum.push({n:this.network[i].getRestoreState(),
                s:this.network[i].getSize(),
                a:this.network[i].weights ? this.network[i].activation : '',
                p:this.network[i].name});
        }
        return sum;
    }

    fit(input, expected, learning_rate) {
        var start = Date.now();

        var prediction = this.input(input).predict();

        var error = [];
        for(var i=0;i<prediction.length;i++) {
            var e1 = prediction[i] - expected[i];
            error.push(e1);
        }

        for(var i=this.network.length-1;i>0;i--) {
            error = this.network[i].propagate(this.network[i-1], error, learning_rate);
        }
        this.totalTime += Date.now() - start;
        return this;
    }

    predict() {
        var res = this.network[0].project(this.inputs);
        for(var i=1;i<this.network.length;i++) {
            res = this.network[i].compute(res);
        }
        return this.network[this.network.length-1].getValues();
    }
}