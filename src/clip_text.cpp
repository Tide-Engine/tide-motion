// Adapted from OpenAI/CLIP d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.
// MIT: licenses/OPENAI-CLIP.txt. Native implementation, no Python runtime.
#include "clip_text.hpp"
#include <ufbx.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <unordered_map>
namespace motion {
namespace {
struct Range {uint32_t first,last;unsigned kind;};
struct Case {uint32_t cp;const char* lower;};
struct Decomp {uint32_t cp;uint32_t chars[4];unsigned count;};
struct Combining {uint32_t cp;unsigned value;};
struct Composition {uint32_t a,b,c;};
struct Entity {std::string_view name,value;};
#include "generated/clip-unicode.inc"
std::string utf8(uint32_t c) {
    std::string s;if(c<128)s+=static_cast<char>(c);
    else if(c<2048){s+=static_cast<char>(192|(c>>6));s+=static_cast<char>(128|(c&63));}
    else if(c<65536){s+=static_cast<char>(224|(c>>12));s+=static_cast<char>(128|((c>>6)&63));s+=static_cast<char>(128|(c&63));}
    else {s+=static_cast<char>(240|(c>>18));s+=static_cast<char>(128|((c>>12)&63));s+=static_cast<char>(128|((c>>6)&63));s+=static_cast<char>(128|(c&63));}return s;
}
std::vector<uint32_t> codepoints(std::string_view s) {
    std::vector<uint32_t> out;
    for(size_t i=0;i<s.size();) {
        auto c=static_cast<unsigned char>(s[i++]);unsigned n=0;uint32_t cp=c,minimum=0;
        if(c>=240&&c<245){n=3;cp=c&7;minimum=65536;}else if(c>=224&&c<240){n=2;cp=c&15;minimum=2048;}else if(c>=194&&c<224){n=1;cp=c&31;minimum=128;}else if(c>=128)throw std::runtime_error("CLIP text is not valid UTF-8");
        if(i+n>s.size())throw std::runtime_error("CLIP text is not valid UTF-8");
        for(unsigned k=0;k<n;++k){auto b=static_cast<unsigned char>(s[i++]);if((b&192)!=128)throw std::runtime_error("CLIP text is not valid UTF-8");cp=(cp<<6)|(b&63);}
        if(cp<minimum||cp>0x10ffff||(cp>=0xd800&&cp<=0xdfff))throw std::runtime_error("CLIP text is not valid UTF-8");
        out.push_back(cp);
    }
    return out;
}
template<class T,size_t N> const T* find(const T (&table)[N],uint32_t c) {
    auto p=std::lower_bound(std::begin(table),std::end(table),c,[](const T& a,uint32_t b){return a.cp<b;});return p!=std::end(table)&&p->cp==c?p:nullptr;
}
unsigned category(uint32_t c){auto p=std::upper_bound(std::begin(unicode_ranges),std::end(unicode_ranges),c,[](uint32_t a,const Range& b){return a<b.first;});if(p==std::begin(unicode_ranges))return 0;--p;return c<=p->last?p->kind:0;}
unsigned combining(uint32_t c){auto p=find(unicode_combining,c);return p?p->value:0;}
uint32_t compose(uint32_t a,uint32_t b) {
    if(a>=0x1100&&a<0x1113&&b>=0x1161&&b<0x1176)return 0xac00+(a-0x1100)*588+(b-0x1161)*28;
    if(a>=0xac00&&a<0xd7a4&&(a-0xac00)%28==0&&b>0x11a7&&b<0x11c3)return a+b-0x11a7;
    auto p=std::lower_bound(std::begin(unicode_composition),std::end(unicode_composition),std::pair{a,b},[](const Composition& c,auto key){return std::pair{c.a,c.b}<key;});return p!=std::end(unicode_composition)&&p->a==a&&p->b==b?p->c:0;
}
std::string nfc(std::string_view text) {
    std::vector<uint32_t> chars;
    for(auto c:codepoints(text)) {
        if(c>=0xac00&&c<0xd7a4){auto s=c-0xac00;chars.push_back(0x1100+s/588);chars.push_back(0x1161+(s%588)/28);if(s%28)chars.push_back(0x11a7+s%28);}
        else if(auto d=find(unicode_decomp,c))chars.insert(chars.end(),d->chars,d->chars+d->count);else chars.push_back(c);
    }
    for(size_t i=1;i<chars.size();++i){size_t j=i;auto cc=combining(chars[j]);while(j&&cc&&combining(chars[j-1])>cc){std::swap(chars[j],chars[j-1]);--j;}}
    std::vector<uint32_t> out;size_t starter=0;unsigned last=0;
    for(auto c:chars){unsigned cc=combining(c);uint32_t composed=out.empty()?0:compose(out[starter],c);if(composed&&(last<cc||last==0))out[starter]=composed;else{if(cc==0)starter=out.size();out.push_back(c);last=cc;}}
    std::string result;for(auto c:out)result+=utf8(c);return result;
}
std::string unescape(std::string_view text) {
    std::string out;
    for(size_t i=0;i<text.size();) {
        if(text[i]!='&'){out+=text[i++];continue;}
        bool found=false;
        if(i+2<text.size()&&text[i+1]=='#') {
            size_t k=i+2;unsigned base=10;if(k<text.size()&&(text[k]=='x'||text[k]=='X')){base=16;++k;}size_t first=k;uint32_t c=0;
            for(;k<text.size();++k){char x=text[k];unsigned digit=x>='0'&&x<='9'?static_cast<unsigned>(x-'0'):x>='a'&&x<='f'?static_cast<unsigned>(x-'a'+10):x>='A'&&x<='F'?static_cast<unsigned>(x-'A'+10):99;if(digit>=base)break;c=std::min<uint32_t>(0x110000,c*base+digit);}
            if(k>first){if(k<text.size()&&text[k]==';')++k;if(c==0||c>0x10ffff||(c>=0xd800&&c<=0xdfff))c=0xfffd;out+=utf8(c);i=k;found=true;}
        } else {
            for(size_t n=std::min<size_t>(32,text.size()-i-1);n>0;--n){auto key=text.substr(i+1,n);auto p=std::lower_bound(std::begin(html_entities),std::end(html_entities),key,[](const Entity& e,std::string_view s){return e.name<s;});if(p!=std::end(html_entities)&&p->name==key){out+=p->value;i+=n+1;found=true;break;}}
        }
        if(!found)out+=text[i++];
    }
    return out;
}
}
struct ClipTokenizer::Impl {
    std::array<std::string,256> bytes;std::unordered_map<std::string,int64_t> vocabulary;
    std::map<std::pair<std::string,std::string>,size_t> ranks;
    explicit Impl(const fs::path& file) {
        std::ifstream in(file,std::ios::binary);std::vector<unsigned char> gzip((std::istreambuf_iterator<char>(in)),{});
        if(gzip.size()<20||gzip.size()>2*1024*1024||gzip[0]!=31||gzip[1]!=139||gzip[2]!=8)throw std::runtime_error("Invalid CLIP BPE archive");
        size_t offset=10;unsigned flags=gzip[3];if(flags&4){if(offset+2>=gzip.size())throw std::runtime_error("Invalid CLIP gzip header");size_t length=gzip[offset]+256*gzip[offset+1];offset+=2+length;}
        for(unsigned flag:{8U,16U})if(flags&flag){while(offset<gzip.size()&&gzip[offset])++offset;++offset;}
        if(flags&2)offset+=2;
        if(offset>=gzip.size()-8)throw std::runtime_error("Invalid CLIP gzip header");
        std::string content(4*1024*1024,'\0');ufbx_inflate_input input{};input.data=gzip.data()+offset;input.data_size=gzip.size()-offset-8;input.total_size=input.data_size;input.no_header=true;input.no_checksum=true;ufbx_inflate_retain retain{};
        auto count=ufbx_inflate(content.data(),content.size(),&input,&retain);if(count<0)throw std::runtime_error("Cannot inflate CLIP BPE archive");content.resize(static_cast<size_t>(count));
        std::vector<unsigned> order;for(unsigned b=33;b<=126;++b)order.push_back(b);for(unsigned b=161;b<=172;++b)order.push_back(b);for(unsigned b=174;b<=255;++b)order.push_back(b);
        unsigned extra=256;for(unsigned b=0;b<256;++b)if(std::find(order.begin(),order.end(),b)==order.end()){order.push_back(b);bytes[b]=utf8(extra++);}else bytes[b]=utf8(b);
        int64_t id=0;for(auto b:order)vocabulary[bytes[b]]=id++;for(auto b:order)vocabulary[bytes[b]+"</w>"]=id++;
        std::istringstream lines(content);std::string line;std::getline(lines,line);
        for(size_t rank=0;rank<48894;++rank){if(!std::getline(lines,line))throw std::runtime_error("CLIP BPE vocabulary truncated");std::istringstream parts(line);std::string a,b;parts>>a>>b;if(a.empty()||b.empty())throw std::runtime_error("Invalid CLIP BPE merge");ranks[{a,b}]=rank;vocabulary[a+b]=id++;}
        vocabulary["<|startoftext|>"]=id++;vocabulary["<|endoftext|>"]=id++;if(id!=49408)throw std::runtime_error("Invalid CLIP vocabulary size");
    }
    std::array<int64_t,77> encode(const std::string& input) const {
        if(input.size()>65536)throw std::runtime_error("CLIP prompt exceeds 64 KiB");
        auto chars=codepoints(unescape(unescape(nfc(input))));std::string lowered;
        for(size_t i=0;i<chars.size();++i){auto c=chars[i];if(c==0x3a3&&i&&category(chars[i-1])==1&&(i+1==chars.size()||category(chars[i+1])!=1))lowered+=utf8(0x3c2);else if(auto p=find(unicode_lower,c))lowered+=p->lower;else lowered+=utf8(c);}
        chars=codepoints(lowered);std::vector<std::string> words;
        for(size_t i=0;i<chars.size();) {
            if(category(chars[i])==3){++i;continue;}size_t end=i+1;
            std::string rest;for(size_t j=i;j<std::min(chars.size(),i+16);++j)rest+=utf8(chars[j]);bool special=false;
            for(const auto* token:{"<|startoftext|>","<|endoftext|>","'s","'t","'re","'ve","'m","'ll","'d"})if(rest.starts_with(token)){end=i+std::string_view(token).size();special=true;break;}
            unsigned kind=category(chars[i]);if(!special&&kind!=2)while(end<chars.size()&&category(chars[end])==kind)++end;
            std::string word;for(size_t j=i;j<end;++j)word+=utf8(chars[j]);words.push_back(std::move(word));i=end;
        }
        std::vector<int64_t> ids{49406};
        for(const auto& word:words) {
            if(word=="<|startoftext|>"||word=="<|endoftext|>"){ids.push_back(vocabulary.at(word));continue;}
            std::vector<std::string> pieces;for(unsigned char b:word)pieces.push_back(bytes[b]);pieces.back()+="</w>";
            while(pieces.size()>1) {
                size_t best=ranks.size();std::pair<std::string,std::string> pair;
                for(size_t j=0;j+1<pieces.size();++j){auto p=ranks.find({pieces[j],pieces[j+1]});if(p!=ranks.end()&&p->second<best){best=p->second;pair=p->first;}}
                if(best==ranks.size())break;
                std::vector<std::string> merged;for(size_t j=0;j<pieces.size();++j){if(j+1<pieces.size()&&pieces[j]==pair.first&&pieces[j+1]==pair.second){merged.push_back(pieces[j]+pieces[j+1]);++j;}else merged.push_back(pieces[j]);}pieces=std::move(merged);
            }
            for(const auto& piece:pieces)ids.push_back(vocabulary.at(piece));
            if(ids.size()>77)break;
        }
        ids.push_back(49407);if(ids.size()>77){ids.resize(77);ids.back()=49407;}
        std::array<int64_t,77> result{};std::copy(ids.begin(),ids.end(),result.begin());return result;
    }
};
ClipTokenizer::ClipTokenizer(const fs::path& p):impl_(std::make_unique<Impl>(p)){}
ClipTokenizer::~ClipTokenizer()=default;
std::array<int64_t,77> ClipTokenizer::encode(const std::string& s) const{return impl_->encode(s);}
}
