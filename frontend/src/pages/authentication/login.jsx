import { Link } from 'react-router-dom';

// material-ui
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import Button from '@mui/material/Button';

// project import
import AuthWrapper from './AuthWrapper';
import AuthLogin from './auth-forms/AuthLogin';

// ================================|| LOGIN ||================================ //

export default function Login() {
  return (
    <AuthWrapper>
      <Grid container spacing={3} justifyContent="center" alignItems="center">
        <Grid item xs={12} textAlign="center">
          <Typography variant="h3">Welcome to SafeOps</Typography>
        </Grid>
        <Grid item>
          <Button variant="contained" color="primary" component={Link} to="/">
            Start Surgery
          </Button>
        </Grid>
      </Grid>
    </AuthWrapper>
  );
}
