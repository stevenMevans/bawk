import logo from './logo.svg';
import './App.css';
import React, {useState} from 'react';
import { Button } from '@material-ui/core';
// import axios from 'axios'

function App() {
    const [showLoading, setShowLoading] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [getMessage, setGetMessage] = useState({})
    const [selectedFile, setSelectedFile] = useState();
    const [isFilePicked, setIsFilePicked] = useState(false);
    const [showSubmit, setShowSubmit] = useState(false);

    const wakeword = "fourth brain";

    const changeHandler = (event) => {
		setSelectedFile(event.target.files[0]);
		setIsFilePicked(true);
		setShowSubmit(true);
	};

	const handleSubmission = () => {
	    setShowSubmit(false);
	    setShowLoading(true);

        const formData = new FormData();
        formData.append("keyword", wakeword);
        formData.append('audio', selectedFile);

        fetch(
			'http://0.0.0.0:5000/api/detect',
			{
			 method: 'POST',
				body: formData,
			}
		 )
        .then((response) => response.json())
        .then((result) => {
            console.log('Success:', result);
            setGetMessage(result);
            setShowLoading(false);
            setShowResults(true);
        })
        .catch((error) => {
            console.error('Error:', error);
        });
    };

	const Loading = () => (
	     <div><h3>Detecting. . .</h3></div>
    )

    const Results = () => (
        <div><h3>Keyword {getMessage.detected ? 'WAS' : 'NOT'} detected</h3></div>
    )

  return (
    <div className="App">
      <header className="App-header">
        <img src={logo} className="App-logo" alt="logo" />
        <h3>Detecting wake word: "{wakeword}"</h3>
        <div>{isFilePicked ?
            null
            :
            <Button
            variant="contained"
            component="label"
            >
              Choose .wav File
              <input
                accept="audio/*"
                type="file"
                hidden
                onChange={changeHandler}
              />
            </Button>}
        </div>


          { showLoading ? <Loading /> : null }
          { showResults ? <Results /> : null }

          <div>{showSubmit ?
              <Button
              variant="contained"
              component="label"
              onClick={handleSubmission}
              >
                Submit Audio
              </Button>
              :
                null}
          </div>

      </header>
    </div>
  );
}

export default App;