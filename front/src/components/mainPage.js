import React from 'react';
import {Switch , Route} from 'react-router-dom'
import AboutPage  from './aboutPage.js'
import ContactPage  from './contactPage.js'
import MainApplicationPage  from './mainApplicationPage.js'

const Main = () => (
    <Switch>
        <Route exact path="/" component={AboutPage} />
        <Route path="/mainApplication" component={MainApplicationPage} />
        <Route path="/contact" component={ContactPage} />   
    </Switch>
)


export default Main;