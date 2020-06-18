import React, {Component} from 'react'
import {Modal } from 'react-bootstrap'
import axios from 'axios'




class ClassificationComponent extends Component {
    constructor(props) {
      super(props);
      this.state = {
        title: '',
        content: '',
        titleError: true,
        contentError: true
      };
     
    }

    state = {
      show: false,
      label:"real"
    }


    handleModal = () => {


      if( this.state.title === '' ||  this.state.content === '' ){
        if(this.state.content === ''){
          this.setState({contentError: false});
        }
        if(this.state.title === ''){
          this.setState({titleError: false});
        }
        if(this.state.content === ''){
          this.setState({contentError: false});
        }
      }else{
        let axiosConfig = {
          headers: {
              'Content-Type': 'application/json;charset=UTF-8',
              "Access-Control-Allow-Origin": "*",
          }
        };
        axios.post(
          'http://127.0.0.1:5000/detector',
          {
            headline: this.state.title,
            body: this.state.content
          }, 
          axiosConfig)
        .then((res) => {
          console.log("RESPONSE RECEIVED: ", res);
          this.setState({
            label: res.data.result === '[1]' ? "real" : "fake",
            show: !this.state.show,
          })
        })
        .catch((err) => {
          alert("Error While Connecting to  http://127.0.0.1:5000/detector ...")
          console.log("AXIOS ERROR: ", err);
        })
      }
    }

    handleChangeTitle = (event) => {
      this.setState({title: event.target.value});
    }

    handleChangeContent = (event) => {
      this.setState({content: event.target.value});
    }

    handleClose = () => {
      this.setState({show: false});
    }

    handleBlur = (event) => {
      if(event.target.id === "title"){
        this.handleChangeTitle(event)
        if(event.target.value !== ""){
          this.setState({titleError: true});
        }
      }
      if(event.target.id === "content"){
        this.handleChangeContent(event)
        if(event.target.value !== ""){
          this.setState({contentError: true});
        }
      }
    }
  
    render() {
        return(
          <div>
              <div className="classificationForm">
                <div className="articleTitle">
                  <label >Article's Title</label>
                  <input className="form-control inputTitle"  id="title" value={this.state.title} onChange={this.handleChangeTitle}  onBlur={this.handleBlur} placeholder="Enter Title" />
                  <span className="text-danger error-message" hidden={this.state.titleError}>Please, enter the title of the article.</span>
                </div>
                <div className="articleContent">
                  <label >Article's Content</label>
                  <textarea className="form-control contentTextarea"  id="content" value={this.state.content} onChange={this.handleChangeContent} onBlur={this.handleBlur}  placeholder="Enter Content"></textarea>
                  <span className="text-danger error-message" hidden={this.state.contentError}>Please, enter the content of the article.</span>
                </div>
                <button type="submit" className="btn btn-lg classificationButton"   onClick={this.handleModal}>Classify the Article</button>
             </div>

             <Modal show={this.state.show} onHide={this.handleClose} className="modalContent">
               <Modal.Header closeButton className="modalTitle">
                 <div>Result of classification</div>
               </Modal.Header>
              <Modal.Body>
                <div>
                  <div className="label">This article is {this.state.label}</div>
                  <div className="icons">
                    {this.state.label === "real" ? 
                       <i className="far fa-check-circle  fa-5x text-success"></i>
                       :
                       <i className="far fa-times-circle  fa-5x text-danger"></i>
                    }
                  </div>
                </div> 
              </Modal.Body>
             </Modal>
          </div>


        )
    }
}

export default ClassificationComponent ;


