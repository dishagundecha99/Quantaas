import CloudDownloadIcon from '@mui/icons-material/CloudDownload';
import RefreshIcon from '@mui/icons-material/Refresh';
import SearchIcon from '@mui/icons-material/Search';
import { Button, Grid, IconButton, InputAdornment, Paper, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField } from '@mui/material';
import React, { useEffect, useState } from 'react';
import axiosConfig from '../utils/AxiosConfig';

const CompletedStats = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [tableData, setTableData] = useState([]);
  const [bestRunData, setBestRunData] = useState("");

  const user_id = localStorage.getItem("user_id");

  const fetchExpDetails = async () => {
    try {
      const response = (await axiosConfig.get('/prev-runs', { params: { "user-id": user_id } })).data;
      console.log('Response:', response);

      setBestRunData(response?.best_runs[Object.keys(response?.best_runs)[0]]);

      let experimentRunDetails = [];

      Object.keys(response?.runs).forEach(function (key) {
        experimentRunDetails.push(response?.runs[key].map((data) => (
          {
            exp_id: data.exp_id,
            name: key,
            learning_rate: data?.hyperparams?.learning_rate,
            weight_decay: data?.hyperparams?.weight_decay,
            batch_size: data?.hyperparams?.batch_size,
            max_epochs: data?.hyperparams?.max_epochs,
            warmup_steps: data?.hyperparams?.warmup_steps
          }
        )));
      });
      setTableData(experimentRunDetails.flat());
      console.log("Completed run table: ", experimentRunDetails.flat(), "Best run: ", response?.best_runs);

    } catch (error) {
      console.error('Error fetching trade details:', error);
    }
  };

  useEffect(() => {
    fetchExpDetails();
  }, []);

  const handleSearch = (event) => {
    setSearchTerm(event.target.value);
  };

  const filteredData = tableData.filter((data) =>
    // TODO: change based on experiment name
    data.exp_id.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const downloadModelFile = async (exp_id) => {
    await axiosConfig.request({
      url: '/download-model/'+exp_id,
      method: 'GET',
      params: { "user-id": user_id },
      responseType: 'blob',
    }).then((response) => {
      const href = URL.createObjectURL(response.data);
      const link = document.createElement('a');
      link.href = href;
      link.setAttribute('download', 'model');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(href);
    }).catch((error) => {
      console.log(error);
    });
  };

  return (
    <div>
      <Grid container spacing={2} pt={2} pb={2} sx={{ height: "50%" }}>
        <Grid item xs={6} sm={6} >
          <TextField
            label="Search"
            variant="outlined"
            size='small'
            value={searchTerm}
            sx={{ width: '100%' }}
            InputProps={{
              endAdornment: (
                <InputAdornment>
                  <IconButton>
                    <SearchIcon />
                  </IconButton>
                </InputAdornment>
              )
            }}
            onChange={handleSearch}
          />
        </Grid>
        <Grid container item xs={6} sm={6} justifyContent={"flex-end"}>
          <Button startIcon={<RefreshIcon />} onClick={fetchExpDetails}>
            REFRESH
          </Button>
        </Grid>
      </Grid>
      <Grid container spacing={2}>
        <Grid item xs={12} sm={12}>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Experiment Name</TableCell>
                  <TableCell>Experiment ID</TableCell>
                  <TableCell>Learning Rate</TableCell>
                  <TableCell>Weight Decay</TableCell>
                  <TableCell>Batch Size</TableCell>
                  <TableCell>Max Epochs</TableCell>
                  <TableCell>Warmup Steps</TableCell>
                  <TableCell>Download</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {filteredData.length === 0 ?
                  <TableRow>
                    <TableCell colSpan={8} sx={{ bgcolor: 'lightgray', fontStyle: "italic" }} align="center">No data found</TableCell>
                  </TableRow>
                  :
                  filteredData.map((data) => (
                    <TableRow key={data.exp_id} sx={{backgroundColor: (data.exp_id === bestRunData) ? "lightgreen" : "inherit"}}>
                      <TableCell>{data.name}</TableCell>
                      <TableCell>{data.exp_id}</TableCell>
                      <TableCell>{data.learning_rate}</TableCell>
                      <TableCell>{data.weight_decay}</TableCell>
                      <TableCell>{data.batch_size}</TableCell>
                      <TableCell>{data.max_epochs}</TableCell>
                      <TableCell>{data.warmup_steps}</TableCell>
                      <TableCell>
                        {
                          <CloudDownloadIcon sx={{ color: "green" }}
                            onClick={() => downloadModelFile(data.exp_id)}
                          />
                        }
                      </TableCell>
                    </TableRow>
                  ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Grid>
      </Grid>
    </div>
  );
};


export default CompletedStats;
