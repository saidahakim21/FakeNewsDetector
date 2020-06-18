import React from 'react';
import './App.css';
import {Content, Navigation, Drawer, Header, Layout} from "react-mdl"
import Main from './components/mainPage.js'
import {Link} from 'react-router-dom'

function App() {
  return (
      <div>
        <Layout fixedHeader >
          <Header className="header" title={<span className="headerTitle" ></span>}>
              <Navigation>
                <Link className="Link" to="/" ><span className="fas fa-bars"> About</span></Link>
                <Link className="Link"  to="/mainApplication"><span className="fas fa-cogs"> Main Application</span></Link>
                <Link className="Link"  to="/contact"><span className="fas fa-users"> Contacts</span></Link>
              </Navigation>
          </Header>
          <Content className="overflow-hidden styleX">
            <Main>
            </Main>
          </Content>
      </Layout>
      </div>
  );
}

export default App;
