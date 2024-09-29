import { useState } from 'react';

// material-ui
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';
import React, { useEffect } from 'react';
import io from 'socket.io-client';

// project import
import MainCard from 'components/MainCard';

const socket = io.connect("http://10.91.189.113:5000");


// ==============================|| DEFAULT - VIDEO CONTAINER ||============================== //

export default function VideoContainer() {

  const [frame, setFrame] = useState(null);

  useEffect(() => {
    // Request video stream from the server
    socket.emit('start_video');

    // Receive frames from the server
    socket.on('frame', (data) => {
      setFrame(`data:image/jpeg;base64,${data.data}`);
    });

    // Cleanup on component unmount
    return () => {
      socket.disconnect();
    };
  }, []);


  return (
    <>
      <Grid container alignItems="center" justifyContent="space-between">
        <Grid item>
          <Typography variant="h5">Live Feed</Typography>
        </Grid>
      </Grid>
      <MainCard content={false} sx={{ mt: 1.5 }}>
        <Box sx={{ pt: 1, pl: 1, pr: 1 }} >
          {/* Video component will be placed here */}
          {frame ? (
            <img src={frame} alt="Video stream" style={{ width: '620px', height: '460px', objectFit: 'cover', justifyContent: 'center' }} />
          ) : (
            <p>Loading...</p>
          )}
        </Box>
      </MainCard>
    </>
  );
}
