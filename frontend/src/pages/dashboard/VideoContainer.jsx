import { useState } from 'react';

// material-ui
import Button from '@mui/material/Button';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Box from '@mui/material/Box';

// project import
import MainCard from 'components/MainCard';

// ==============================|| DEFAULT - VIDEO CONTAINER ||============================== //

export default function VideoContainer() {

  return (
    <>
      <Grid container alignItems="center" justifyContent="space-between">
        <Grid item>
          <Typography variant="h5">Live Feed</Typography>
        </Grid>
      </Grid>
      <MainCard content={false} sx={{ mt: 1.5 }}>
        <Box sx={{ pt: 1, pl: 1, pr: 1 }}>
          {/* Video component will be placed here */}
          <video width="100%" controls>
            <source src="your-video-url-here.mp4" type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        </Box>
      </MainCard>
    </>
  );
}
