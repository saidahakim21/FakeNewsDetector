import React, {Component} from 'react'


class Contact extends Component {


    handleClick = () => {
        window.open("https://github.com/saidahakim21/FakeNewsDetector");
    }
    render() {
        return(
            <div className="container">
                <div className="contact">
                    <span className="fa-stack fa-4x contactIcon">
                        <i className="fas fa-circle fa-stack-2x "></i>
                        <i className="fas fa-users fa-stack-1x fa-inverse usersIcon"></i>
                    </span>
                    <div className="contactText">Members</div>
                    <div>
                        <div className="membersText">           
                            This project was carried out by:
                        </div>
                        <div className="membersInfo">
                            <div className="member">
                                <span className="nameMember">Abdelhakim SAID</span>
                                <div > <i class="far fa-envelope"></i> ahakim.said@gmail.com</div>
                            </div>
                            <div className="member">
                                <span className="nameMember">Darine ATHMANI</span>
                                <div ><i class="far fa-envelope"></i> athdarine@gmail.com</div>
                            </div>
                            <div className="member">
                                <span className="nameMember">Safia ZIHMOU</span>
                                <div className="mailMember"> <i class="far fa-envelope"></i> safiazihmou@gmail.com</div>
                            </div>
                            <div className="member">
                                <span className="nameMember">Yacine ZABAT</span>
                                <div className="mailMember"><i class="far fa-envelope"></i> zabatyacine32@gmail.com</div>
                            </div>          
                        </div>
                        <div className="membersText">
                            Students in Master 1 Systèmes Intelligents Communicants (SIC) 
                            <div>Cergy-Paris University</div>
                        </div>
                    </div>
                </div>
                <div className="github">
                    <span className="fa fa-github fa-8x  contactIcon "></span>
                    <div className="contactText">Github Repository</div>
                    <div>
                        <div className="githubText">
                            This project is available and can be downloaded from our github repository :
                        </div>
                        <div >
                            <div className="linkGithub" onClick={this.handleClick}><i class="fas fa-link"></i> Fake news Detector Repositpry</div>
                        </div> 
                    </div>
                </div>
            </div>
        )
    }
}
export default Contact;

 