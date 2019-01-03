#include "Proxy.h"
#include "util/Util.h"
#include <regex>

using namespace std;

void Proxy::init(const std::string& param){
    vector<string> strLayer = getStringList(param, "-");
    nLayer = strLayer.size();
    // raw string: R"(...)"
    string srShape = R"((\d+)(\*(\d+))*)";
    //regex ri(srShape); // input layer
    regex ra(R"((\d+),a,(sigmoid|relu|tanh))"); // activation layer
    regex rc(R"((\d+),c,()"+srShape+")"); // convolutional layer
    regex rp(R"((\d+),p,(max|mean|min),()"+srShape+")"); // pooling layer
    iShape[0] = oShape[0] = getShape(strLayer[0]);
    typeLayer[0] = LayerType::Input;
    for(size_t i = 1; i<strLayer.size(); ++i){
        smatch m;
        if(regex_match(strLayer[i], m, ra)){
            iShape[i] = {1};
            oShape[i] = {1};
            nNodeLayer[i] = stoi(m[1]);
            if(m[2] == "sigmoid")
                typeLayer[i] = LayerType::Sigmoid;
            else if(m[2] == "relu")
                typeLayer[i] = LayerType::Relu;
            else if(m[2] == "tanh")
                typeLayer[i] = LayerType::Tanh;
        }else if(regex_match(strLayer[i], m, rc)){
            nNodeLayer[i] = stoi(m[1]);
            typeLayer[i] = LayerType::Conv;
            vector<int> shape = getShape(m[2]);
        }else if(regex_match(strLayer[i], m, rp)){
            nNodeLayer[i] = stoi(m[1]);
            typeLayer[i] = LayerType::Pool;
            string type = m[2];
            vector<int> shape = getShape(m[3]);
        }
    }
}
    // std::vector<LayerType> ltype;
    // std::vector<vector<int>> iShape, oShape; // shape of input and output of each layer
    
std::vector<int> Proxy::getShape(const string& str){
    return getIntList(str, "*");
}

int Proxy::getSize(const std::vector<int>& shape){
    if(shape.empty())
        return 0;
    int r = 1;
    for(auto& v : shape)
        r*=v;
    return r;
}
