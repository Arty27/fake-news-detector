import {
  Box,
  Chip,
  Grid,
  LinearProgress,
  Paper,
  Typography,
} from "@mui/material";

const CustomGrid = ({ factor, data }) => {
  return (
    <Grid item xs={12} md={6} key={factor}>
      <Paper
        elevation={0}
        sx={{
          p: 2,
          width: "300px",
          border: "1px solid",
          borderColor: "divider",
          borderRadius: 2,
        }}
      >
        <Box
          sx={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            mb: 1,
          }}
        >
          <Typography
            variant="subtitle1"
            sx={{
              fontWeight: 600,
              textTransform: "capitalize",
            }}
          >
            {factor.replace("_", " ")}
          </Typography>
          <Chip
            label={`${((data?.score || 0) * 100).toFixed(1)}%`}
            color={
              (data?.score || 0) > 0.7
                ? "error"
                : (data?.score || 0) > 0.4
                ? "warning"
                : "success"
            }
            size="small"
          />
        </Box>
        <LinearProgress
          variant="determinate"
          value={(data?.score || 0) * 100}
          color={
            (data?.score || 0) > 0.7
              ? "error"
              : (data?.score || 0) > 0.4
              ? "warning"
              : "success"
          }
          sx={{ height: 6, borderRadius: 3, mb: 1 }}
        />
        <Box
          sx={{
            // display: "block",
            // justifyContent: "space-between",
            fontSize: "0.875rem",
          }}
        >
          <div>Weight: {((data?.weight || 0) * 100).toFixed(0)}%</div>

          <div>
            Contribution: {((data?.contribution || 0) * 100).toFixed(1)}%
          </div>
        </Box>
        {data?.decision && (
          <Chip
            label={data.decision}
            variant="outlined"
            size="small"
            sx={{ mt: 1 }}
          />
        )}
      </Paper>
    </Grid>
  );
};

export default CustomGrid;
