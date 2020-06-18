import React, {Component} from 'react'
import {Modal } from 'react-bootstrap'
import axios from 'axios'


class IdentificationComponent extends Component {
    constructor(props) {
        super(props);
        this.state = {
          sentence: '',
          sentenceError: true,
          show: false,
        };
     
    }

    state = {
        trueInformation: undefined,
        uri1: "",
        uri2: "",
        relation: "",
    }

    handleChangeSentence = (event) => {
        this.setState({sentence: event.target.value});
    }

    handleBlur = (event) => {
        this.handleChangeSentence(event)
        if(event.target.value !== ""){
            this.setState({sentenceError: true});
        }
    }

    handleResult = () => {
        if( this.state.sentence === ''){
            this.setState({sentenceError: false});
        }else{

            this.setState({show: false});
            let axiosConfig = {
                headers: {
                    'Content-Type': 'application/json;charset=UTF-8',
                    "Access-Control-Allow-Origin": "*",
                }
            };

            let url = "http://127.0.0.1:5000/identifier?text="+this.state.sentence

            axios.get(
                url, 
                axiosConfig
            ).then((res) => {
                console.log("RESPONSE RECEIVED: ", res);
                console.log(res.data)

                if(res.data.result){
                    /* the information is true */
                    this.setState({
                        trueInformation: true,
                        show: !this.state.show
                    })
                }else if (res.data.uri_1 && res.data.uri_2 && res.data.rel){
                    this.setState({
                        trueInformation: false,
                        uri1: res.data.uri_1,
                        uri2: res.data.uri_2,
                        relation: res.data.rel,
                        show: !this.state.show
                    })
                }else{
                    console.log("rien")
                }

            }).catch((err) => {
                alert("Error While Connecting to  http://127.0.0.1:5000/identifier ...")
                console.log("AXIOS ERROR: ", err);
            })


            
        }
    }

    render() {
        return(
            <div>
                <div className="classificationForm">
                    <div className="sentence">
                    <label >Sentence</label>
                    <input className="form-control inputSentence"  id="sentence" value={this.state.sentence} onChange={this.handleChangeSentence}  onBlur={this.handleBlur} placeholder="Enter Sentence" />
                    <span className="text-danger error-message" hidden={this.state.sentenceError}>Please, enter the title of the article.</span>
                    </div>
                    <button type="submit" className="btn btn-lg classificationButton"   onClick={this.handleResult}>Identifcation</button>
                    {this.state.show &&  (
                        <div>
                            <div className="sentence"> Results: </div>
                            <div className="resultsBox">
                                {this.state.trueInformation
                                ?
                                    <div className="resultLabel bold"> <i className="far fa-check-circle  fa-1x text-success"></i>  This information is Real.</div>
                                :
                                    <div>
                                        <div className="resultLabel bold"> <i className="far fa-times-circle  fa-1x text-danger"></i>  This information is Fake.</div>
                                        <div className="resultsDetails">
                                            <div><span className="resultLabel">The relation:  </span><span className="resultValue">{this.state.relation}</span><span className="resultLabel"> between</span></div> 
                                            <div><span className="resultLabel">URI 1: </span><span className="resultValue">{this.state.uri1}</span></div>
                                            <div><span className="resultLabel">URI 2: </span><span className="resultValue">{this.state.uri2}</span>  <span className="resultLabel text-danger bold" >  is FAKE</span></div>
                                            
                                        </div>
                                    </div>
                                }
                            </div>
                        </div>
                    )
                    }
                </div>
            </div>
            
        )
    }
}

export default IdentificationComponent ;


