import React, {Component} from 'react'
import {Link, Navigation} from 'react-router-dom'
import logo from './logo.png';



class About extends Component {
    render() {
        return(
            <div className="aboutLayout">
                <div className="aboutLogo">
                    <center><img  src={logo} alt="Logo" /></center>
                </div>
                <div>
                    <div className="descriptionBox">
                            <div className="descriptionContent">
                            This is a project that has been developed to fight against false information.
                            <div className="description">It can be used for:</div>
                            <div><i class="fas fa-angle-right"></i>Determine if an item is fake or real.</div>
                            <div><i class="fas fa-angle-right"></i>Identify in case an article is fake the false information contained in it.</div>
                            </div>
                            <div>  
                                <div className="start">
                                    <Link to="/mainApplication"><i className="fa fa-play fa-6x startIcon"></i></Link>
                                    <Link to="/mainApplication" className="startText"><div>Start Verification</div></Link>
                                </div>
                            </div>
                            
                    </div>
                </div>
                
            </div>
        )
    }
}

export default About;