import React, { useState } from "react";
import { makeStyles } from '@material-ui/core/styles';
import { CircularProgress, IconButton } from "@material-ui/core";
import { green } from '@material-ui/core/colors';
import MicIcon from '@material-ui/icons/Mic';
import MicOffIcon from '@material-ui/icons/MicOff';

const useStyles = makeStyles((theme) => ({
  wrapper: {
    margin: theme.spacing(1),
    position: 'relative',
  },
  micButton: {
    zIndex: 2,
  },
  fabProgress: {
    color: green[500],
    position: 'absolute',
    top: -6,
    left: -6,
    zIndex: 1,
  },
  buttonProgress: {
    color: green[500],
    position: 'absolute',
    top: -8,
    left: '47.25%',
    zIndex: 1
  }
}));

const AudioRecoder = ({ handleSubmit }) => {
    const STREAM_ERROR_MESSAGE = 'MediaDevices.getUserMedia() threw an error. Stream did not open.';
    const classes = useStyles();
    const [mediaRecorder, setMediaRecorder] = useState(null);
    const [recording, setRecording] = useState(false);
    const [recordedChunks, setRecordedChunks] = useState([]);

    const startRecording = async () => {
        setRecording(true);

        const stream = await getAudioStream();
        const recorder = getMediaRecorder(stream);

        recorder.addEventListener('dataavailable', onDataAvailable);
        recorder.addEventListener('stop', onMediaRecorderStop);

        recorder.start();
        return recorder;
    }

    const stopRecording = () => {
        setRecording(false);
        if (mediaRecorder) {
            mediaRecorder.stop();
        }
        setMediaRecorder(null);
    }

    const getAudioStream = async () => {
        let stream;

        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: false, audio: true
            });
        } catch (error) {
            throw new Error(
                `${STREAM_ERROR_MESSAGE} ${error.name} - ${error.message}`
            );
        }

        return stream;
    }

    const getMediaRecorder = (stream) => {
        let recorder = mediaRecorder;
        if (!recorder) {
            recorder = new MediaRecorder(stream);
            setMediaRecorder(recorder);
        }
        return recorder;
    }

    const onDataAvailable = ({ data }) => {
        if (data.size > 0) {
            recordedChunks.push(data);
        }
    }

    const onMediaRecorderStop = () => {
        const blob = new Blob(recordedChunks);
        const file = new File([blob], `temp.wav`, {
            type: 'audio/wav;',
            lastModified: Date.now()
        });
        handleSubmit(file);
    }

    return (
        <div>
            {
                recording
                    ?
                    <div className={classes.wrapper}>

                        <IconButton className={classes.micButton} onClick={stopRecording} color="primary" aria-label="microphone on" component="span">
                            <MicIcon/>
                        </IconButton>
                        <CircularProgress size={68} className={classes.buttonProgress} />
                        <h6>Press microphone to stop recording and detect wake word</h6>
                    </div>
                    :
                    <div className={classes.wrapper}>

                        <IconButton onClick={startRecording} color="secondary" aria-label="microphone off" component="span">
                            <MicOffIcon/>
                        </IconButton>
                        <h6>Press microphone to record audio</h6>
                    </div>
            }
        </div>
    );
}

export default AudioRecoder;