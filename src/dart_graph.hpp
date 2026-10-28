#pragma once
// Minimal ONNX protobuf construction for generated native operator declarations.
// All learned initializer data remains in the separately verified tensor file.
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
// Graph construction is a cold installer path. Avoid spending optimizer time
// expanding thousands of std::string operations; inference graphs are unchanged.
#if defined(__clang__)
#define DART_GRAPH_BUILD [[clang::optnone, gnu::noinline]]
#elif defined(__GNUC__)
#define DART_GRAPH_BUILD [[gnu::optimize("O0"), gnu::noinline]]
#else
#define DART_GRAPH_BUILD
#endif
inline void dart_varint(std::string& out,uint64_t value) {
    while(value>=128){out.push_back(static_cast<char>((value&127)|128));value>>=7;}
    out.push_back(static_cast<char>(value));
}
inline std::string dv(unsigned field,uint64_t value) {
    std::string out;dart_varint(out,static_cast<uint64_t>(field)*8);dart_varint(out,value);return out;
}
inline std::string db(unsigned field,std::string_view bytes) {
    std::string out;dart_varint(out,static_cast<uint64_t>(field)*8+2);dart_varint(out,bytes.size());out.append(bytes);return out;
}
inline std::string df(unsigned field,uint64_t bits,unsigned size) {
    std::string out;dart_varint(out,static_cast<uint64_t>(field)*8+(size==4?5:1));
    for(unsigned i=0;i<size;++i)out.push_back(static_cast<char>((bits>>(8*i))&255));
    return out;
}
inline std::string dr(std::initializer_list<unsigned char> bytes) {
    if(bytes.size()==0)return {};
    return {reinterpret_cast<const char*>(bytes.begin()),bytes.size()};
}
inline std::string dz(size_t count){return std::string(count,'\0');}
inline std::string dc(size_t width) {
    std::string out(width*width*4,'\0');
    for(size_t r=0;r<width;++r)for(size_t c=r+1;c<width;++c){auto i=(r*width+c)*4;out[i+2]=static_cast<char>(128);out[i+3]=static_cast<char>(255);}
    return out; // F32 causal mask: zero on/below diagonal, -infinity above.
}
inline std::string dn(std::string_view op,std::string_view name,std::initializer_list<std::string_view> inputs,std::initializer_list<std::string_view> outputs,std::initializer_list<std::string> attributes) {
    std::string out;
    for(auto v:inputs)out+=db(1,v);
    for(auto v:outputs)out+=db(2,v);
    if(!name.empty())out+=db(3,name);
    out+=db(4,op);
    for(const auto& v:attributes)out+=db(5,v);
    return db(1,out);
}
