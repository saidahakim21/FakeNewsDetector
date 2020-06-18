import React, {Component} from 'react'
import ClassificationComponent from './classificationComponent.js'
import IdentificationComponent from './identificationComponent.js'
import { Tab, Tabs, TabList, TabPanel } from 'react-tabs';
import 'react-tabs/style/react-tabs.css';


class MainApplication extends Component {
    render() {
        return(
            <div>
                <div className="main">
                    <Tabs>
                        <TabList>
                        <Tab>Classification</Tab>
                        <Tab>Identification</Tab>
                        </TabList>
                        <TabPanel>
                            <ClassificationComponent/>
                        </TabPanel>
                        <TabPanel>
                            <IdentificationComponent/>
                        </TabPanel>
                    </Tabs>
                </div>
                
            </div>
        )
    }
}

export default MainApplication;