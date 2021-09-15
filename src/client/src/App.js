import React, {useState} from 'react';
import {makeStyles} from '@material-ui/core/styles';
import TextField from '@material-ui/core/TextField';
import AudioRecoder from "./AudioRecoder";
import {
    AppBar,
    CircularProgress,
    Container,
    FormControl,
    Grid,
    InputLabel,
    MenuItem,
    Paper,
    Select,
    Toolbar,
    Typography
} from "@material-ui/core";
import CheckCircleOutlineIcon from '@material-ui/icons/CheckCircleOutline';
import HighlightOffIcon from '@material-ui/icons/HighlightOff';
import WarningIcon from '@material-ui/icons/Warning';
import InfoIcon from '@material-ui/icons/Info';

const useStyles = makeStyles((theme) => ({
    root: {
        flexGrow: 1,
    },
    paper: {
        // height: 500,
        padding: theme.spacing(2),
        textAlign: 'center',
        // width: 100,
    },
    control: {
        padding: theme.spacing(2),
    },
    formControl: {
        margin: theme.spacing(1),
        minWidth: 120,
    },
    selectEmpty: {
        marginTop: theme.spacing(2),
    },
    results: {
        padding: theme.spacing(2),
        textAlign: 'center',
        height: 30,
        border: '2 solid red',
        borderRadius: 5
    }
}));

function App() {
    const MODEL_ENDPOINT_MAP = {
        1: 'custom',
        2: 'pretrained'
    }
    const DEFAULT_WAKE_WORD = 'Hey Fourth Brain';
    const DEFAULT_MODEL = 1;
    const classes = useStyles();
    const [showInitial, setShowInitial] = useState(true);
    const [showLoading, setShowLoading] = useState(false);
    const [showResults, setShowResults] = useState(false);
    const [showFailure, setShowFailure] = useState(false);
    const [getMessage, setGetMessage] = useState({});
    const [wakeWord, setWakeWord] = useState(DEFAULT_WAKE_WORD);
    const [model, setModel] = useState(DEFAULT_MODEL);

    const handleSubmission = (file) => {
        setShowLoading(true);
        setShowResults(false);
        setShowInitial(false);
        setShowFailure(false);

        const formData = new FormData();
        formData.append("keyword", wakeWord);
        formData.append('audio', file);

        fetch(
            `http://ec2-34-222-238-222.us-west-2.compute.amazonaws.com:5000/api/${MODEL_ENDPOINT_MAP[model]}`,
            {
                method: 'POST',
                body: formData,
            }
        )
            .then((response) => response.json())
            .then((result) => {
                console.log('Success:', result);
                if (result.status === 'Success') {
                    setGetMessage(result);
                    setShowLoading(false);
                    setShowResults(true);
                } else {
                    setShowFailure(true);
                    setShowLoading(false);
                    setShowResults(false);
                }
            })
            .catch((error) => {
                console.error('Error:', error);
                setShowFailure(true);
                setShowLoading(false);
                setShowResults(false);
            });
    };

    const Loading = () => (
        <div>
            <CircularProgress/>
            <h3>Detecting. . .</h3>
        </div>
    )

    const Results = () => (
        <div>
            {getMessage.detected ? <CheckCircleOutlineIcon/> : <HighlightOffIcon/>}
            <h3>"{wakeWord}" {getMessage.detected ? 'WAS' : 'NOT'} detected</h3>
        </div>
    )

    const Failure = () => (
        <div>
            <WarningIcon />
            <h3>An error occurred while processing the audio</h3>
        </div>
    )

    const Initial = () => (
        <div>
            <InfoIcon />
            <h3>Use the microphone to record audio and detect wake words</h3>
        </div>
    )


    const handleWakeWordChange = (event) => {
        setWakeWord(event.target.value);
    }

    const handleModelChange = (event) => {
        console.log(event.target.value);
        setModel(event.target.value);
    }

    return (
        <div className="App">
            <Container>
                <Paper>
                    <AppBar position="static">
                        <Toolbar>
                            <Typography variant="h4">
                                Custom Wake Word Detection
                            </Typography>
                        </Toolbar>
                    </AppBar>
                    <Grid container className={classes.root} spacing={2}>
                        <Grid item xs={6}>
                            <Typography className={classes.paper}>
                                <FormControl className={classes.formControl}>
                                    <InputLabel shrink>
                                        Model
                                    </InputLabel>
                                    <Select
                                        value={model}
                                        onChange={handleModelChange}
                                    >
                                        <MenuItem value={1}>Custom</MenuItem>
                                        <MenuItem value={2}>Pretrained</MenuItem>
                                    </Select>
                                </FormControl>
                            </Typography>
                        </Grid>
                        <Grid item xs={6}>
                            <Typography className={classes.paper}>
                                <TextField id="custom-wake-word" label="Wake Word" defaultValue={DEFAULT_WAKE_WORD}
                                           onChange={handleWakeWordChange}/>
                            </Typography>
                        </Grid>
                        <Grid item xs={12}>
                            <Typography className={classes.paper}>
                                <AudioRecoder handleSubmit={handleSubmission}/>
                            </Typography>
                        </Grid>
                        <Grid item xs={12}>
                            <Typography className={classes.paper}>
                                {showInitial && <Initial/>}
                                {showLoading && <Loading/>}
                                {showResults && <Results/>}
                                {showFailure && <Failure/>}
                            </Typography>
                        </Grid>
                    </Grid>
                </Paper>
            </Container>
        </div>
    );
}

export default App;