import { useState } from "react";
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Chip,
  LinearProgress,
  Paper,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from "@mui/material";
import { ExpandMore, Info } from "@mui/icons-material";
import {
  Shield,
  AlertTriangle,
  CheckCircle2,
  XCircle,
  BarChart3,
  Target,
  Globe,
  User,
} from "lucide-react";
import CustomGrid from "./CustomGrid";

const ResultsDisplay = ({ results }) => {
  const [expanded, setExpanded] = useState("panel1");

  // Safety check - if no results, show message
  if (!results) {
    return (
      <Box sx={{ textAlign: "center", py: 4 }}>
        <Typography variant="h6" color="text.secondary">
          No analysis results available
        </Typography>
      </Box>
    );
  }

  // Safety check - if results don't have required properties, show message
  if (!results.final_verdict || !results.fake_news_score) {
    return (
      <Box sx={{ textAlign: "center", py: 4 }}>
        <Typography variant="h6" color="text.secondary">
          Incomplete analysis results
        </Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
          Some required data is missing from the analysis
        </Typography>
      </Box>
    );
  }

  const handleAccordionChange = (panel) => (event, isExpanded) => {
    setExpanded(isExpanded ? panel : false);
  };

  const getVerdictColor = (verdict) => {
    const lowerVerdict = verdict.toLowerCase();
    if (lowerVerdict.includes("fake")) return "error";
    if (lowerVerdict.includes("suspicious")) return "warning";
    if (lowerVerdict.includes("real")) return "success";
    return "info";
  };

  const getVerdictIcon = (verdict) => {
    const lowerVerdict = verdict.toLowerCase();
    if (lowerVerdict.includes("fake")) return <XCircle size={24} />;
    if (lowerVerdict.includes("suspicious")) return <AlertTriangle size={24} />;
    if (lowerVerdict.includes("real")) return <CheckCircle2 size={24} />;
    return <Info size={24} />;
  };

  const getConfidenceColor = (confidence) => {
    const lowerConfidence = confidence.toLowerCase();
    if (lowerConfidence === "high") return "success";
    if (lowerConfidence === "medium") return "warning";
    return "error";
  };

  const getRiskLevelColor = (riskLevel) => {
    const lowerRisk = riskLevel.toLowerCase();
    if (lowerRisk === "low") return "success";
    if (lowerRisk === "medium") return "warning";
    return "error";
  };

  const formatTimestamp = (timestamp) => {
    if (typeof timestamp === "string") {
      return new Date(timestamp).toLocaleString();
    }
    return "N/A";
  };

  return (
    <Box>
      <Typography variant="h5" gutterBottom sx={{ fontWeight: 600, mb: 3 }}>
        <BarChart3
          size={24}
          style={{ marginRight: 8, verticalAlign: "middle" }}
        />
        Analysis Results
      </Typography>

      {/* Main Verdict Card */}
      <Card
        sx={{
          mb: 3,
          border: "2px solid",
          borderColor: `${getVerdictColor(results.final_verdict)}.main`,
        }}
      >
        <CardContent sx={{ p: 3 }}>
          <Grid container spacing={3} alignItems="center">
            <Grid item xs={12} md={8}>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}
              >
                {getVerdictIcon(results.final_verdict)}
                <Typography
                  variant="h4"
                  component="h2"
                  sx={{ fontWeight: 700 }}
                >
                  {results.final_verdict}
                </Typography>
              </Box>
              <Typography variant="body1" color="text.secondary" sx={{ mb: 2 }}>
                {results.reasoning}
              </Typography>
              <Box sx={{ display: "flex", gap: 2, flexWrap: "wrap" }}>
                <Chip
                  label={`Confidence: ${results.confidence_level || "Unknown"}`}
                  color={getConfidenceColor(
                    results.confidence_level || "Unknown"
                  )}
                  variant="outlined"
                  size="medium"
                />
                {results.processing_time_ms && (
                  <Chip
                    label={`Processing Time: ${results.processing_time_ms}ms`}
                    color="info"
                    variant="outlined"
                    size="medium"
                  />
                )}
                {results.analysis_timestamp && (
                  <Chip
                    label={`Analysis Time: ${formatTimestamp(
                      results.analysis_timestamp
                    )}`}
                    color="info"
                    variant="outlined"
                    size="medium"
                  />
                )}
              </Box>
            </Grid>
            <Grid item xs={12} md={4}>
              <Box sx={{ textAlign: "center" }}>
                <Typography
                  variant="h2"
                  component="div"
                  sx={{
                    fontWeight: 700,
                    color: `${getVerdictColor(results.final_verdict)}.main`,
                  }}
                >
                  {(results.fake_news_score * 100).toFixed(1)}%
                </Typography>
                <Typography variant="body2" color="text.secondary">
                  Fake News Score
                </Typography>
                <LinearProgress
                  variant="determinate"
                  value={results.fake_news_score * 100}
                  color={getVerdictColor(results.final_verdict)}
                  sx={{ mt: 2, height: 8, borderRadius: 4 }}
                />
              </Box>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      {/* Article Information (for URL analysis) */}
      {(results.article_source || results.article_headline) && (
        <Card sx={{ mb: 3, backgroundColor: "grey.50" }}>
          <CardContent sx={{ p: 3 }}>
            <Typography
              variant="h6"
              gutterBottom
              sx={{ fontWeight: 600, mb: 2 }}
            >
              📰 Article Information
            </Typography>
            <Grid container spacing={2}>
              {results.article_source && (
                <Grid item xs={12} md={6}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 1,
                    }}
                  >
                    <Typography variant="subtitle2" color="text.secondary">
                      🏢 Source:
                    </Typography>
                  </Box>
                  <Typography variant="body1" sx={{ fontWeight: 600 }}>
                    {results.article_source}
                  </Typography>
                </Grid>
              )}
              {results.article_headline && (
                <Grid item xs={12} md={6}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 1,
                    }}
                  >
                    <Typography variant="subtitle2" color="text.secondary">
                      📝 Headline:
                    </Typography>
                  </Box>
                  <Typography variant="body1" sx={{ fontWeight: 600 }}>
                    {results.article_headline}
                  </Typography>
                </Grid>
              )}
              {results.article_url && (
                <Grid item xs={12}>
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      gap: 1,
                      mb: 1,
                    }}
                  >
                    <Typography variant="subtitle2" color="text.secondary">
                      🔗 URL:
                    </Typography>
                  </Box>
                  <Typography variant="body2" color="primary.main">
                    <a
                      href={results.article_url}
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ textDecoration: "none" }}
                    >
                      {results.article_url}
                    </a>
                  </Typography>
                </Grid>
              )}
            </Grid>
          </CardContent>
        </Card>
      )}

      {/* Dashboard Data */}
      {results.dashboard_data && (
        <Grid container spacing={3} sx={{ mb: 3 }}>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: "center" }}>
                <Box
                  sx={{
                    width: 60,
                    height: 60,
                    borderRadius: "50%",
                    backgroundColor:
                      results.dashboard_data?.color || "primary.main",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    mx: "auto",
                    mb: 2,
                  }}
                >
                  <Shield size={32} color="white" />
                </Box>
                <Typography variant="h6" gutterBottom>
                  Risk Level
                </Typography>
                <Chip
                  label={(
                    results.dashboard_data?.risk_level || "unknown"
                  ).toUpperCase()}
                  color={getRiskLevelColor(
                    results.dashboard_data?.risk_level || "unknown"
                  )}
                  size="large"
                  sx={{ fontWeight: 600 }}
                />
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: "center" }}>
                <Box
                  sx={{
                    width: 60,
                    height: 60,
                    borderRadius: "50%",
                    backgroundColor: "primary.main",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    mx: "auto",
                    mb: 2,
                  }}
                >
                  <Target size={32} color="white" />
                </Box>
                <Typography variant="h6" gutterBottom>
                  Verdict Category
                </Typography>
                <Typography
                  variant="body1"
                  sx={{ fontWeight: 600, textTransform: "capitalize" }}
                >
                  {(
                    results.dashboard_data?.verdict_category || "unknown"
                  ).replace("_", " ")}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
          <Grid item xs={12} md={4}>
            <Card>
              <CardContent sx={{ textAlign: "center" }}>
                <Box
                  sx={{
                    width: 60,
                    height: 60,
                    borderRadius: "50%",
                    backgroundColor: "secondary.main",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    mx: "auto",
                    mb: 2,
                  }}
                >
                  <BarChart3 size={32} color="white" />
                </Box>
                <Typography variant="h6" gutterBottom>
                  Total Score
                </Typography>
                <Typography
                  variant="h4"
                  sx={{ fontWeight: 700, color: "secondary.main" }}
                >
                  {results.fake_news_score.toFixed(3)}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      )}

      {/* Factor Breakdown */}
      {results.factor_breakdown &&
        Object.keys(results.factor_breakdown).length > 0 && (
          <Accordion
            expanded={expanded === "panel1"}
            onChange={handleAccordionChange("panel1")}
            sx={{
              mb: 2,
            }}
          >
            <AccordionSummary expandIcon={<ExpandMore />}>
              <Typography variant="h6" sx={{ fontWeight: 600 }}>
                <BarChart3
                  size={20}
                  style={{ marginRight: 8, verticalAlign: "middle" }}
                />
                Factor Breakdown
              </Typography>
            </AccordionSummary>
            <AccordionDetails>
              <Grid container spacing={2}>
                {Object.entries(results.factor_breakdown).map(
                  ([factor, data]) => (
                    <CustomGrid factor={factor} data={data} />
                  )
                )}
              </Grid>
            </AccordionDetails>
          </Accordion>
        )}

      {/* Live News Verification Results */}
      {results.live_match_results && (
        <Accordion
          expanded={expanded === "panel2"}
          onChange={handleAccordionChange("panel2")}
          sx={{ mb: 2 }}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              <Globe
                size={20}
                style={{ marginRight: 8, verticalAlign: "middle" }}
              />
              Live News Verification
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box sx={{ mb: 3 }}>
              <Box
                sx={{ display: "flex", alignItems: "center", gap: 2, mb: 2 }}
              >
                <Chip
                  label={
                    results.factor_breakdown.live_checker.decision || "Unknown"
                  }
                  color={
                    results.factor_breakdown.live_checker.decision?.includes(
                      "strong"
                    )
                      ? "success"
                      : results.factor_breakdown.live_checker.decision?.includes(
                          "partial"
                        )
                      ? "warning"
                      : results.factor_breakdown.live_checker.decision?.includes(
                          "no corroboration"
                        )
                      ? "error"
                      : "info"
                  }
                  size="medium"
                  sx={{ fontWeight: 600 }}
                />
                {results.factor_breakdown.live_checker.verification_score && (
                  <Chip
                    label={`Score: ${(
                      results.factor_breakdown.live_checker.verification_score *
                      100
                    ).toFixed(1)}%`}
                    variant="outlined"
                    size="medium"
                  />
                )}
                {results.factor_breakdown.live_checker.queries_generated && (
                  <Chip
                    label={`${results.factor_breakdown.live_checker.queries_generated} Queries`}
                    variant="outlined"
                    size="medium"
                  />
                )}
              </Box>
              {!results.factor_breakdown.live_checker.decision?.includes(
                "no corroboration"
              ) &&
                results.live_match_results &&
                results.live_match_results.length > 0 && (
                  <Box>
                    <Typography
                      variant="h6"
                      gutterBottom
                      sx={{ fontWeight: 600, mb: 2 }}
                    >
                      📰 Matching Articles Found
                    </Typography>
                    <Grid container spacing={2}>
                      {results.live_match_results.map((match, index) => (
                        <Grid item xs={12} key={index}>
                          <Paper
                            elevation={0}
                            sx={{
                              p: 2,
                              width: "80vw",
                              border: "1px solid",
                              borderColor: "divider",
                              borderRadius: 2,
                              backgroundColor: "background.paper",
                              "&:hover": {
                                backgroundColor: "grey.50",
                                borderColor: "primary.main",
                              },
                              transition: "all 0.2s ease",
                            }}
                          >
                            <Box
                              sx={{
                                display: "flex",
                                justifyContent: "space-between",
                                alignItems: "flex-start",
                                mb: 1,
                              }}
                            >
                              <Box sx={{ flex: 1 }}>
                                <Typography
                                  variant="subtitle1"
                                  sx={{ fontWeight: 600, mb: 1 }}
                                >
                                  {match.title || "No Title"}
                                </Typography>
                                <Typography
                                  variant="body2"
                                  color="text.secondary"
                                  sx={{ mb: 1 }}
                                >
                                  🏢 <strong>Source:</strong>{" "}
                                  {match.source || "Unknown Source"}
                                </Typography>
                                {match.url && (
                                  <Typography
                                    variant="body2"
                                    color="primary.main"
                                    sx={{ mb: 1 }}
                                  >
                                    🔗{" "}
                                    <a
                                      href={match.url}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      style={{ textDecoration: "none" }}
                                    >
                                      {match.url}
                                    </a>
                                  </Typography>
                                )}
                                {match.published_at && (
                                  <Typography
                                    variant="body2"
                                    color="text.secondary"
                                    sx={{ fontSize: "0.875rem" }}
                                  >
                                    📅{" "}
                                    {new Date(
                                      match.published_at
                                    ).toLocaleDateString()}
                                  </Typography>
                                )}
                              </Box>
                              <Box sx={{ textAlign: "right", ml: 2 }}>
                                <Chip
                                  label={`${(match.similarity * 100).toFixed(
                                    1
                                  )}% Match`}
                                  color={
                                    match.similarity > 0.7
                                      ? "success"
                                      : match.similarity > 0.4
                                      ? "warning"
                                      : "error"
                                  }
                                  size="small"
                                  sx={{ fontWeight: 600, mb: 1 }}
                                />
                                <Typography
                                  variant="caption"
                                  color="text.secondary"
                                  display="block"
                                >
                                  Rank #{index + 1}
                                </Typography>
                              </Box>
                            </Box>
                          </Paper>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                )}
            </Box>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Named Entities Found */}
      {results.named_entities && results.named_entities.length > 0 && (
        <Accordion
          expanded={expanded === "panel3"}
          onChange={handleAccordionChange("panel3")}
          sx={{ mb: 2 }}
        >
          <AccordionSummary expandIcon={<ExpandMore />}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              <User
                size={20}
                style={{ marginRight: 8, verticalAlign: "middle" }}
              />
              Named Entities Identified
            </Typography>
          </AccordionSummary>
          <AccordionDetails>
            <Box>
              <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                The following entities were identified in the article using
                AI-powered Named Entity Recognition:
              </Typography>
              <Grid container spacing={2}>
                {results.named_entities.map((entity, index) => (
                  <Grid item xs={12} sm={6} md={4} key={index}>
                    <Paper
                      elevation={0}
                      sx={{
                        p: 2,
                        border: "1px solid",
                        borderColor: "divider",
                        borderRadius: 2,
                        textAlign: "center",
                        backgroundColor: "background.paper",
                        "&:hover": {
                          backgroundColor: "grey.50",
                          borderColor: "primary.main",
                        },
                        transition: "all 0.2s ease",
                      }}
                    >
                      <Box
                        sx={{
                          width: 40,
                          height: 40,
                          borderRadius: "50%",
                          backgroundColor: "primary.main",
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          mx: "auto",
                          mb: 1,
                        }}
                      >
                        <Typography
                          variant="body2"
                          color="white"
                          sx={{ fontWeight: 600 }}
                        >
                          {entity.label?.charAt(0) || "E"}
                        </Typography>
                      </Box>
                      <Typography
                        variant="subtitle2"
                        sx={{ fontWeight: 600, mb: 0.5 }}
                      >
                        {entity.text || "Unknown Entity"}
                      </Typography>
                      <Chip
                        label={entity.label || "Unknown Type"}
                        color="primary"
                        size="small"
                        variant="outlined"
                      />
                    </Paper>
                  </Grid>
                ))}
              </Grid>
            </Box>
          </AccordionDetails>
        </Accordion>
      )}

      {/* Raw Data */}
      {/* <Accordion
        expanded={expanded === "panel4"}
        onChange={handleAccordionChange("panel4")}
      >
        <AccordionSummary expandIcon={<ExpandMore />}>
          <Typography variant="h6" sx={{ fontWeight: 600 }}>
            <Info
              size={20}
              style={{ marginRight: 8, verticalAlign: "middle" }}
            />
            Raw Analysis Data
          </Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Paper
            elevation={0}
            sx={{ p: 2, backgroundColor: "grey.50", borderRadius: 2 }}
          >
            <pre
              style={{
                fontSize: "0.875rem",
                overflow: "auto",
                whiteSpace: "pre-wrap",
                fontFamily: "monospace",
              }}
            >
              {JSON.stringify(results, null, 2)}
            </pre>
          </Paper>
        </AccordionDetails>
      </Accordion> */}
    </Box>
  );
};

export default ResultsDisplay;
